// Copyright 2024 ByteDance and/or its affiliates.
// Copyright 2026 Basalt SchurEKF integration contributors.
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along
// with this program. If not, see <https://www.gnu.org/licenses/>.

#include <basalt/vi_estimator/schur_vins_ekf.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <basalt/camera/stereographic_param.hpp>
#include <basalt/utils/assert.h>
#include <basalt/utils/system_utils.h>

namespace basalt {

namespace {
constexpr double kNsToSec = 1e-9;
constexpr double kMinDepth = 1e-6;
constexpr double kMinDt = 1e-9;
}  // namespace

SchurVinsEkfEstimator::SchurVinsEkfEstimator(
    const Eigen::Vector3d& g, const Calibration<double>& calib,
    const VioConfig& config)
    : calib_(calib),
      config_(config),
      gravity_(g),
      max_regular_frames_(std::max(1, config.vio_schur_max_frames)),
      max_kfs_(std::max(1, config.vio_max_kfs)),
      focal_length_(focalLength()) {
  BASALT_ASSERT_MSG(calib_.intrinsics.size() >= 2 && calib_.T_i_c.size() >= 2,
                    "SCHUR_EKF backend requires stereo camera calibration.");
  initNoise();
  initCovariance();
  vision_data_queue.set_capacity(10);
  imu_data_queue.set_capacity(300);
}

SchurVinsEkfEstimator::~SchurVinsEkfEstimator() {
  if (processing_thread_) {
    imu_data_queue.push(nullptr);
    vision_data_queue.push(nullptr);
  }
  maybe_join();
}

void SchurVinsEkfEstimator::initialize(
    int64_t t_ns, const Sophus::SE3d& T_w_i, const Eigen::Vector3d& vel_w_i,
    const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
  initializeState(t_ns, T_w_i, vel_w_i, bg, ba);
  initialize(bg, ba);
}

void SchurVinsEkfEstimator::initialize(const Eigen::Vector3d& bg,
                                       const Eigen::Vector3d& ba) {
  if (processing_thread_) return;
  auto proc_func = [this, bg, ba] { processingLoop(bg, ba); };
  processing_thread_.reset(new std::thread(proc_func));
}

void SchurVinsEkfEstimator::maybe_join() {
  if (processing_thread_) {
    processing_thread_->join();
    processing_thread_.reset();
  }
}

void SchurVinsEkfEstimator::setMaxStates(size_t val) {
  max_regular_frames_ = std::max<size_t>(1, val);
}

void SchurVinsEkfEstimator::setMaxKfs(size_t val) {
  max_kfs_ = std::max<size_t>(1, val);
}

void SchurVinsEkfEstimator::addIMUToQueue(const ImuData<double>::Ptr& data) {
  imu_data_queue.emplace(data);
}

void SchurVinsEkfEstimator::addVisionToQueue(
    const OpticalFlowResult::Ptr& data) {
  vision_data_queue.push(data);
}

void SchurVinsEkfEstimator::debug_finalize() {
  std::cout << "=== SchurEKF stats all ===\n";
  stats_all_.print();
  std::cout << "=== SchurEKF stats sums ===\n";
  stats_sums_.print();
}

void SchurVinsEkfEstimator::processingLoop(Eigen::Vector3d bg,
                                           Eigen::Vector3d ba) {
  ImuData<double>::Ptr imu = popFromImuDataQueue();
  if (!imu) {
    finished = true;
    if (out_vis_queue) out_vis_queue->push(nullptr);
    if (out_marg_queue) out_marg_queue->push(nullptr);
    if (out_state_queue) out_state_queue->push(nullptr);
    return;
  }
  calibrateImu(*imu);

  Vec3 last_acc = imu->accel;
  Vec3 last_gyro = imu->gyro;

  while (true) {
    OpticalFlowResult::Ptr curr_frame;
    vision_data_queue.pop(curr_frame);

    if (config_.vio_enforce_realtime) {
      while (!vision_data_queue.empty()) vision_data_queue.pop(curr_frame);
    }

    if (!curr_frame) break;

    if (!initialized_) {
      while (imu && imu->t_ns < curr_frame->t_ns) {
        last_acc = imu->accel;
        last_gyro = imu->gyro;
        imu = popFromImuDataQueue();
        if (imu) calibrateImu(*imu);
      }

      Sophus::SE3d T_w_i;
      T_w_i.setQuaternion(Eigen::Quaterniond::FromTwoVectors(last_acc,
                                                             Vec3::UnitZ()));
      initializeState(curr_frame->t_ns, T_w_i, Vec3::Zero(), bg, ba);

      if (config_.vio_debug || config_.vio_extended_logging) {
        std::cout << "Setting up SchurEKF: t_ns " << curr_frame->t_ns
                  << std::endl;
        std::cout << "T_w_i\n" << current_state_.T_w_i.matrix() << std::endl;
      }
    } else {
      while (imu && imu->t_ns <= curr_frame->t_ns) {
        const double dt = (imu->t_ns - current_state_.t_ns) * kNsToSec;
        if (dt > kMinDt) propagate(dt, last_acc, last_gyro);
        last_acc = imu->accel;
        last_gyro = imu->gyro;
        imu = popFromImuDataQueue();
        if (imu) calibrateImu(*imu);
      }

      const double dt = (curr_frame->t_ns - current_state_.t_ns) * kNsToSec;
      if (dt > kMinDt) propagate(dt, last_acc, last_gyro);
      current_state_.t_ns = curr_frame->t_ns;
    }

    measure(curr_frame);
  }

  if (out_vis_queue) out_vis_queue->push(nullptr);
  if (out_marg_queue) out_marg_queue->push(nullptr);
  if (out_state_queue) out_state_queue->push(nullptr);

  finished = true;
  std::cout << "Finished SchurEKF VIO" << std::endl;
}

ImuData<double>::Ptr SchurVinsEkfEstimator::popFromImuDataQueue() {
  ImuData<double>::Ptr data;
  imu_data_queue.pop(data);
  return data;
}

void SchurVinsEkfEstimator::calibrateImu(ImuData<double>& data) const {
  data.accel = calib_.calib_accel_bias.getCalibrated(data.accel);
  data.gyro = calib_.calib_gyro_bias.getCalibrated(data.gyro);
}

void SchurVinsEkfEstimator::initializeState(int64_t t_ns,
                                            const Sophus::SE3d& T_w_i,
                                            const Vec3& vel_w_i,
                                            const Vec3& bg, const Vec3& ba) {
  current_state_.t_ns = t_ns;
  current_state_.T_w_i = T_w_i;
  current_state_.vel_w_i = vel_w_i;
  current_state_.bg = bg;
  current_state_.ba = ba;
  T_w_i_init_ = T_w_i;
  initialized_ = true;
  initCovariance();
}

void SchurVinsEkfEstimator::initCovariance() {
  cov_ = Eigen::MatrixXd::Zero(15, 15);
  cov_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-4;
  cov_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-3;
  cov_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 1e-3;
  cov_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 2e-3;
  cov_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-6;
}

void SchurVinsEkfEstimator::initNoise() {
  imu_noise_.setZero();
  imu_noise_.block<3, 3>(0, 0) =
      calib_.gyro_noise_std.array().square().matrix().asDiagonal();
  imu_noise_.block<3, 3>(3, 3) =
      calib_.accel_noise_std.array().square().matrix().asDiagonal();
  imu_noise_.block<3, 3>(6, 6) =
      calib_.accel_bias_std.array().square().matrix().asDiagonal();
  imu_noise_.block<3, 3>(9, 9) =
      calib_.gyro_bias_std.array().square().matrix().asDiagonal();
}

void SchurVinsEkfEstimator::propagate(double dt, const Vec3& accel_meas,
                                      const Vec3& gyro_meas) {
  BASALT_ASSERT(dt >= 0);
  if (dt <= kMinDt) return;

  const Vec3 accel = accel_meas - current_state_.ba;
  const Vec3 gyro = gyro_meas - current_state_.bg;
  const Eigen::Matrix3d R = current_state_.T_w_i.so3().matrix();

  Mat15 F = Mat15::Zero();
  Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();

  F.block<3, 3>(0, 12) = -R;
  F.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -skew(R * accel);
  F.block<3, 3>(6, 9) = -R;

  G.block<3, 3>(0, 0) = -R;
  G.block<3, 3>(6, 3) = -R;
  G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
  G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();

  const Mat15 Fdt = F * dt;
  const Mat15 Fdt2 = Fdt * Fdt;
  const Mat15 Fdt3 = Fdt2 * Fdt;
  const Mat15 Phi = Mat15::Identity() + Fdt + 0.5 * Fdt2 + Fdt3 / 6.0;

  predictState(dt, accel, gyro);

  cov_.block(0, 0, 15, 15) =
      Phi * cov_.block(0, 0, 15, 15) * Phi.transpose() +
      Phi * G * imu_noise_ * G.transpose() * Phi.transpose() * dt;

  if (cov_.rows() > 15) {
    const int clone_cols = cov_.cols() - 15;
    cov_.block(0, 15, 15, clone_cols) =
        Phi * cov_.block(0, 15, 15, clone_cols);
    cov_.block(15, 0, clone_cols, 15) =
        cov_.block(15, 0, clone_cols, 15) * Phi.transpose();
  }

  cov_ = (cov_ + cov_.transpose()) * 0.5;
  current_state_.t_ns += static_cast<int64_t>(std::llround(dt * 1e9));
}

void SchurVinsEkfEstimator::predictState(double dt, const Vec3& accel,
                                         const Vec3& gyro) {
  const Eigen::Quaterniond q0 = current_state_.T_w_i.unit_quaternion();
  const Vec3 wd_full = gyro * dt;
  const Vec3 wd_half = wd_full * 0.5;

  Eigen::Quaterniond dq_half(1.0, wd_half.x() * 0.5, wd_half.y() * 0.5,
                             wd_half.z() * 0.5);
  Eigen::Quaterniond dq_full(1.0, wd_full.x() * 0.5, wd_full.y() * 0.5,
                             wd_full.z() * 0.5);
  dq_half.normalize();
  dq_full.normalize();

  const Eigen::Quaterniond q_half = q0 * dq_half;
  const Eigen::Quaterniond q_full = q0 * dq_full;

  const Vec3 k1_dv = q0 * accel + gravity_;
  const Vec3 k1_dp = current_state_.vel_w_i;

  const Vec3 k1_v = current_state_.vel_w_i + k1_dv * dt * 0.5;
  const Vec3 k2_dv = q_half * accel + gravity_;
  const Vec3 k2_dp = k1_v;

  const Vec3 k2_v = current_state_.vel_w_i + k2_dv * dt * 0.5;
  const Vec3 k3_dv = q_half * accel + gravity_;
  const Vec3 k3_dp = k2_v;

  const Vec3 k3_v = current_state_.vel_w_i + k3_dv * dt;
  const Vec3 k4_dv = q_full * accel + gravity_;
  const Vec3 k4_dp = k3_v;

  current_state_.T_w_i.setQuaternion(q_full.normalized());
  current_state_.T_w_i.translation() +=
      dt / 6.0 * (k1_dp + 2.0 * k2_dp + 2.0 * k3_dp + k4_dp);
  current_state_.vel_w_i +=
      dt / 6.0 * (k1_dv + 2.0 * k2_dv + 2.0 * k3_dv + k4_dv);
}

bool SchurVinsEkfEstimator::measure(
    const OpticalFlowResult::Ptr& opt_flow_meas) {
  Timer t_total;
  stats_sums_.add("frame_id", opt_flow_meas->t_ns).format("none");

  augmentClone(opt_flow_meas->t_ns);
  prev_opt_flow_res_[opt_flow_meas->t_ns] = opt_flow_meas;

  int connected0 = 0;
  std::unordered_set<KeypointId> unconnected_obs0;
  std::map<int64_t, int> num_points_connected;
  registerObservations(opt_flow_meas, connected0, unconnected_obs0,
                       num_points_connected);
  const int num_points_added =
      maybeCreateKeyframe(opt_flow_meas, connected0, unconnected_obs0);

  visualUpdate();
  const int removed_observations = removeOutliers(opt_flow_meas);
  const int removed_points = removePointOutliers();
  const WindowManagementStats window_stats = manageWindow(num_points_connected);

  int regular_frames = 0;
  for (const auto& kv : clone_poses_) {
    if (kf_ids_.count(kv.first) == 0) regular_frames++;
  }

  stats_sums_.add("schur_connected0", connected0);
  stats_sums_.add("schur_unconnected0",
                   static_cast<double>(unconnected_obs0.size()));
  stats_sums_.add("schur_new_landmarks", num_points_added);
  stats_sums_.add("schur_num_clones", static_cast<double>(clone_poses_.size()));
  stats_sums_.add("schur_num_kfs", static_cast<double>(kf_ids_.size()));
  stats_sums_.add("schur_num_regular_frames", regular_frames);
  stats_sums_.add("schur_num_landmarks",
                   static_cast<double>(lmdb_.getLandmarks().size()));
  stats_sums_.add("schur_cov_rows", cov_.rows());
  stats_sums_.add("schur_cov_min_diag",
                   cov_.rows() > 0 ? cov_.diagonal().minCoeff() : 0.0);
  stats_sums_.add("schur_removed_observations", removed_observations);
  stats_sums_.add("schur_removed_points", removed_points);
  stats_sums_.add("schur_removed_regular_clones",
                   window_stats.removed_regular_clones);
  stats_sums_.add("schur_removed_keyframes", window_stats.removed_keyframes);

  publishState();
  publishVisualization(opt_flow_meas);

  last_processed_t_ns = current_state_.t_ns;
  stats_sums_.add("schur_measure", t_total.elapsed()).format("ms");
  return true;
}

void SchurVinsEkfEstimator::augmentClone(int64_t t_ns) {
  if (clone_poses_.count(t_ns) > 0) return;

  const int old_size = cov_.rows();
  cov_.conservativeResize(old_size + 6, old_size + 6);
  cov_.block(old_size, 0, 6, old_size) = cov_.block(0, 0, 6, old_size);
  cov_.block(0, old_size, old_size, 6) = cov_.block(0, 0, old_size, 6);
  cov_.block(old_size, old_size, 6, 6) = cov_.block(0, 0, 6, 6);
  cov_ = (cov_ + cov_.transpose()) * 0.5;

  ClonePose clone;
  clone.T_w_i = current_state_.T_w_i;
  clone_poses_[t_ns] = clone;
  updateCloneIndices();
}

void SchurVinsEkfEstimator::updateCloneIndices() {
  int idx = 0;
  for (auto& kv : clone_poses_) kv.second.index = idx++;
}

int SchurVinsEkfEstimator::cloneIndex(int64_t t_ns) const {
  auto it = clone_poses_.find(t_ns);
  if (it == clone_poses_.end()) return -1;
  return it->second.index;
}

void SchurVinsEkfEstimator::registerObservations(
    const OpticalFlowResult::Ptr& opt_flow_meas, int& connected0,
    std::unordered_set<KeypointId>& unconnected_obs0,
    std::map<int64_t, int>& num_points_connected) {
  for (size_t cam_id = 0; cam_id < opt_flow_meas->observations.size(); cam_id++) {
    const TimeCamId tcid_target(opt_flow_meas->t_ns, cam_id);

    for (const auto& kv_obs : opt_flow_meas->observations[cam_id]) {
      const KeypointId kpt_id = kv_obs.first;
      if (lmdb_.landmarkExists(kpt_id)) {
        const TimeCamId& host = lmdb_.getLandmark(kpt_id).host_kf_id;

        KeypointObservation<double> kobs;
        kobs.kpt_id = static_cast<int>(kpt_id);
        kobs.pos = kv_obs.second.translation().cast<double>();
        lmdb_.addObservation(tcid_target, kobs);

        num_points_connected[host.frame_id]++;
        if (cam_id == 0) connected0++;
      } else if (cam_id == 0) {
        unconnected_obs0.emplace(kpt_id);
      }
    }
  }
}

int SchurVinsEkfEstimator::maybeCreateKeyframe(
    const OpticalFlowResult::Ptr& opt_flow_meas, int connected0,
    const std::unordered_set<KeypointId>& unconnected_obs0) {
  const double denom = connected0 + unconnected_obs0.size();
  if (denom > 0.0 &&
      connected0 / denom < config_.vio_new_kf_keypoints_thresh &&
      frames_after_kf_ > config_.vio_min_frames_after_kf) {
    take_kf_ = true;
  }

  if (!take_kf_) {
    frames_after_kf_++;
    return 0;
  }

  take_kf_ = false;
  frames_after_kf_ = 0;
  kf_ids_.emplace(opt_flow_meas->t_ns);
  clone_poses_.at(opt_flow_meas->t_ns).is_keyframe = true;

  int num_points_added = 0;
  for (KeypointId lm_id : unconnected_obs0) {
    if (triangulateLandmark(lm_id, opt_flow_meas)) num_points_added++;
  }
  num_points_kf_[opt_flow_meas->t_ns] = num_points_added;
  return num_points_added;
}

bool SchurVinsEkfEstimator::triangulateLandmark(
    KeypointId lm_id, const OpticalFlowResult::Ptr& opt_flow_meas) {
  if (opt_flow_meas->observations.empty() ||
      opt_flow_meas->observations[0].count(lm_id) == 0) {
    return false;
  }

  std::map<TimeCamId, KeypointObservation<double>> kp_obs;
  for (const auto& kv : prev_opt_flow_res_) {
    for (size_t cam_id = 0; cam_id < kv.second->observations.size(); cam_id++) {
      auto it = kv.second->observations[cam_id].find(lm_id);
      if (it == kv.second->observations[cam_id].end()) continue;

      KeypointObservation<double> kobs;
      kobs.kpt_id = static_cast<int>(lm_id);
      kobs.pos = it->second.translation().cast<double>();
      kp_obs[TimeCamId(kv.first, cam_id)] = kobs;
    }
  }

  const TimeCamId host_tcid(opt_flow_meas->t_ns, 0);
  const Vec2 p0 = opt_flow_meas->observations[0].at(lm_id)
                      .translation()
                      .cast<double>();

  Eigen::Vector4d p0_3d;
  if (!calib_.intrinsics[0].unproject(p0, p0_3d)) return false;

  const double min_triang_distance2 =
      config_.vio_min_triangulation_dist * config_.vio_min_triangulation_dist;

  for (const auto& kv_obs : kp_obs) {
    const TimeCamId& target_tcid = kv_obs.first;
    if (target_tcid == host_tcid) continue;
    if (clone_poses_.count(target_tcid.frame_id) == 0) continue;

    Eigen::Vector4d p1_3d;
    if (!calib_.intrinsics[target_tcid.cam_id].unproject(kv_obs.second.pos,
                                                         p1_3d)) {
      continue;
    }

    const Sophus::SE3d& T_w_i0 = clone_poses_.at(host_tcid.frame_id).T_w_i;
    const Sophus::SE3d& T_w_i1 = clone_poses_.at(target_tcid.frame_id).T_w_i;
    const Sophus::SE3d T_i0_i1 = T_w_i0.inverse() * T_w_i1;
    const Sophus::SE3d T_0_1 = calib_.T_i_c[0].inverse() * T_i0_i1 *
                               calib_.T_i_c[target_tcid.cam_id];

    if (T_0_1.translation().squaredNorm() < min_triang_distance2) continue;

    const Eigen::Vector4d p0_triangulated =
        BundleAdjustmentBase<double>::triangulate(p0_3d.head<3>(),
                                                  p1_3d.head<3>(), T_0_1);

    if (p0_triangulated.array().isFinite().all() && p0_triangulated[3] > 0 &&
        p0_triangulated[3] < 3.0) {
      Keypoint<double> kpt_pos;
      kpt_pos.host_kf_id = host_tcid;
      kpt_pos.direction = StereographicParam<double>::project(p0_triangulated);
      kpt_pos.inv_dist = p0_triangulated[3];
      lmdb_.addLandmark(lm_id, kpt_pos);

      for (const auto& obs : kp_obs) lmdb_.addObservation(obs.first, obs.second);
      point_meta_[lm_id].ekf_init = false;
      point_meta_[lm_id].local_status = true;
      return true;
    }
  }

  return false;
}

void SchurVinsEkfEstimator::visualUpdate() {
  Timer t;
  const int rows = static_cast<int>(clone_poses_.size()) * 6;
  if (rows < 12) {
    stats_sums_.add("schur_update_skipped", 1);
    return;
  }

  const int max_iterations = std::max(1, config_.vio_schur_max_iterations);
  VisualUpdateStats last_stats;
  StateUpdateStats last_update;

  for (int iter = 0; iter < max_iterations; iter++) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(rows, rows);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);
    Eigen::aligned_unordered_map<KeypointId, PointLinData> point_data;
    VisualUpdateStats iter_stats;

    for (const auto& kv : lmdb_.getLandmarks()) {
      iter_stats.candidate_landmarks++;
      PointLinData lin_data;
      if (addLandmarkLinearization(kv.first, kv.second, H, b, lin_data,
                                   iter_stats)) {
        point_data.emplace(kv.first, lin_data);
        iter_stats.schur_landmarks++;
      }
    }

    last_stats = iter_stats;
    last_stats.iterations = iter + 1;
    if (point_data.empty()) break;

    const bool update_points = iter == max_iterations - 1;
    last_update = stateUpdate(H, b, point_data, update_points);
    last_stats.point_updates = last_update.point_updates;
    last_stats.point_skipped = last_update.point_skipped;
    last_stats.current_dx_norm = last_update.current_dx_norm;
    last_stats.clone_dx_norm = last_update.clone_dx_norm;
    if (!last_update.updated) break;
  }

  const double residual_mean =
      last_stats.valid_observations > 0
          ? last_stats.residual_sum / last_stats.valid_observations
          : 0.0;
  stats_sums_.add("schur_update", t.elapsed()).format("ms");
  stats_sums_.add("schur_update_iterations", last_stats.iterations);
  stats_sums_.add("schur_candidate_landmarks",
                   last_stats.candidate_landmarks);
  stats_sums_.add("schur_schur_landmarks", last_stats.schur_landmarks);
  stats_sums_.add("schur_valid_observations", last_stats.valid_observations);
  stats_sums_.add("schur_residual_mean", residual_mean);
  stats_sums_.add("schur_residual_max", last_stats.residual_max);
  stats_sums_.add("schur_current_dx_norm", last_stats.current_dx_norm);
  stats_sums_.add("schur_clone_dx_norm", last_stats.clone_dx_norm);
  stats_sums_.add("schur_point_updates", last_stats.point_updates);
  stats_sums_.add("schur_point_skipped", last_stats.point_skipped);
}

bool SchurVinsEkfEstimator::addLandmarkLinearization(
    KeypointId lm_id, const Keypoint<double>& kpt, Eigen::MatrixXd& H,
    Eigen::VectorXd& b, PointLinData& lin_data,
    VisualUpdateStats& stats) const {
  auto meta_it = point_meta_.find(lm_id);
  if (meta_it != point_meta_.end() && !meta_it->second.local_status) return false;
  if (static_cast<int>(kpt.obs.size()) < config_.vio_schur_min_obs) return false;
  if (!worldPointFromLandmark(kpt, lin_data.p_w)) return false;

  int valid_obs = 0;
  for (const auto& obs : kpt.obs) {
    const int idx = cloneIndex(obs.first.frame_id);
    if (idx < 0) continue;

    Eigen::Vector2d residual;
    Mat26 J_clone;
    Mat23 J_point;
    if (!computeObservation(lin_data.p_w, obs.first, obs.second, residual,
                            &J_clone, &J_point)) {
      continue;
    }

    const double raw_residual_norm = residual.norm();
    stats.residual_sum += raw_residual_norm;
    stats.residual_max = std::max(stats.residual_max, raw_residual_norm);

    const double r2 = residual.squaredNorm();
    double huber_scale = 1.0;
    const double huber_thresh2 = config_.vio_schur_huber_thresh *
                                 config_.vio_schur_huber_thresh;
    if (r2 > huber_thresh2) {
      const double radius = std::sqrt(r2);
      huber_scale = std::sqrt(std::max(std::numeric_limits<double>::min(),
                                       config_.vio_schur_huber_thresh / radius));
      residual *= huber_scale;
      J_clone *= huber_scale;
      J_point *= huber_scale;
    }

    const double obs_invdev = 1.0 / config_.vio_obs_std_dev;
    residual *= obs_invdev;
    J_clone *= obs_invdev;
    J_point *= obs_invdev;

    const int bias = idx * 6;
    H.block<6, 6>(bias, bias).noalias() += J_clone.transpose() * J_clone;
    b.segment<6>(bias).noalias() += J_clone.transpose() * residual;
    lin_data.V.noalias() += J_point.transpose() * J_point;
    lin_data.gv.noalias() += J_point.transpose() * residual;
    auto w_it = lin_data.W_by_clone.find(idx);
    if (w_it == lin_data.W_by_clone.end()) {
      w_it = lin_data.W_by_clone.emplace(idx, Mat63::Zero()).first;
    }
    w_it->second.noalias() += J_clone.transpose() * J_point;
    valid_obs++;
    stats.valid_observations++;
  }

  if (valid_obs < config_.vio_schur_min_obs) return false;

  Eigen::FullPivLU<Eigen::Matrix3d> lu(lin_data.V);
  if (lu.rank() < 3) return false;

  const Eigen::Matrix3d Vinv = lin_data.V.inverse();
  for (auto it_i = lin_data.W_by_clone.begin(); it_i != lin_data.W_by_clone.end(); ++it_i) {
    const int bias_i = it_i->first * 6;
    const Mat63 WVinv = it_i->second * Vinv;
    for (auto it_j = it_i; it_j != lin_data.W_by_clone.end(); ++it_j) {
      const int bias_j = it_j->first * 6;
      const Mat6 schur = WVinv * it_j->second.transpose();
      H.block<6, 6>(bias_i, bias_j) -= schur;
      if (bias_i != bias_j) H.block<6, 6>(bias_j, bias_i) -= schur.transpose();
    }
    b.segment<6>(bias_i) -= WVinv * lin_data.gv;
  }

  return true;
}

bool SchurVinsEkfEstimator::computeObservation(
    const Eigen::Vector3d& p_w, const TimeCamId& tcid,
    const Eigen::Vector2d& obs_px, Eigen::Vector2d& residual, Mat26* J_clone,
    Mat23* J_point) const {
  if (tcid.cam_id >= calib_.intrinsics.size() ||
      tcid.cam_id >= calib_.T_i_c.size()) {
    return false;
  }

  auto clone_it = clone_poses_.find(tcid.frame_id);
  if (clone_it == clone_poses_.end()) return false;

  const Sophus::SE3d& T_w_i = clone_it->second.T_w_i;
  const Sophus::SE3d& T_i_c = calib_.T_i_c[tcid.cam_id];
  const Eigen::Quaterniond q_w_i = T_w_i.unit_quaternion();
  const Eigen::Quaterniond q_i_c = T_i_c.unit_quaternion();
  const Eigen::Matrix3d R_i_c = T_i_c.so3().matrix();
  const Eigen::Vector3d t_i_c = T_i_c.translation();

  const Eigen::Vector3d p_i = q_w_i.inverse() * (p_w - T_w_i.translation());
  const Eigen::Vector3d p_c = q_i_c.inverse() * (p_i - t_i_c);
  if (p_c.z() <= kMinDepth || !p_c.array().isFinite().all()) return false;

  Mat23 pred_jacobian = Mat23::Zero();
  if (config_.vio_schur_use_pixel_residual) {
    Vec2 pred_px;
    if (J_clone || J_point) {
      if (!calib_.intrinsics[tcid.cam_id].project(p_c, pred_px,
                                                   &pred_jacobian)) {
        return false;
      }
    } else if (!calib_.intrinsics[tcid.cam_id].project(p_c, pred_px)) {
      return false;
    }

    if (!pred_px.array().isFinite().all()) return false;
    residual = obs_px - pred_px;
  } else {
    const Eigen::Vector3d obs_bearing = getBearing(tcid, obs_px);
    if (!obs_bearing.array().isFinite().all() ||
        std::abs(obs_bearing.z()) < kMinDepth) {
      return false;
    }

    const Eigen::Vector2d obs_unit = obs_bearing.head<2>() / obs_bearing.z();
    residual = (obs_unit - p_c.head<2>() / p_c.z()) * focal_length_;

    if (J_clone || J_point) {
      const double z2 = p_c.z() * p_c.z();
      pred_jacobian(0, 0) = 1.0 / p_c.z();
      pred_jacobian(1, 1) = 1.0 / p_c.z();
      pred_jacobian(0, 2) = -p_c.x() / z2;
      pred_jacobian(1, 2) = -p_c.y() / z2;
      pred_jacobian *= focal_length_;
    }
  }

  if (J_clone || J_point) {
    const Eigen::Matrix3d R_c_w =
        (q_w_i * q_i_c).toRotationMatrix().transpose();

    if (J_clone) {
      Eigen::Matrix<double, 3, 6> dpc_dpose;
      dpc_dpose.leftCols<3>().noalias() =
          R_i_c.transpose() * skew(p_i) * q_w_i.toRotationMatrix().transpose();
      dpc_dpose.rightCols<3>() = -R_c_w;
      J_clone->noalias() = pred_jacobian * dpc_dpose;
    }

    if (J_point) {
      J_point->noalias() = pred_jacobian * R_c_w;
    }
  }

  return true;
}

SchurVinsEkfEstimator::StateUpdateStats SchurVinsEkfEstimator::stateUpdate(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
    const Eigen::aligned_unordered_map<KeypointId, PointLinData>& point_data,
    bool update_points) {
  StateUpdateStats stats;
  const int rows = H.rows();
  if (rows == 0) return stats;
  BASALT_ASSERT(rows == static_cast<int>(clone_poses_.size()) * 6);
  BASALT_ASSERT(cov_.rows() == rows + 15);

  Eigen::MatrixXd R = (H + H.transpose()) * 0.5;
  R.diagonal().array() += 1e-9;

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(rows, rows + 15);
  J.block(0, 15, rows, rows) = H;

  Eigen::MatrixXd S = H * cov_.bottomRightCorner(rows, rows) * H.transpose() + R;
  S = (S + S.transpose()) * 0.5;
  S.diagonal().array() += 1e-9;

  Eigen::MatrixXd K_T = S.ldlt().solve(H * cov_.bottomRows(rows));
  Eigen::MatrixXd K = K_T.transpose();
  Eigen::VectorXd dX = K * b;

  if (!dX.array().isFinite().all()) return stats;

  const Eigen::VectorXd clone_dx = dX.tail(rows);
  stats.updated = true;
  stats.current_dx_norm = dX.head<15>().norm();
  stats.clone_dx_norm = clone_dx.norm();

  applyCurrentInc(dX.head<15>());

  for (auto& kv : clone_poses_) {
    const int bias = 15 + kv.second.index * 6;
    applyCloneInc(kv.second, dX.segment<6>(bias));
  }

  const Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(K.rows(), K.rows()) - K * J;
  cov_ = I_KH * cov_;
  cov_ = (cov_ + cov_.transpose()) * 0.5;

  if (!update_points) return stats;

  for (const auto& kv : point_data) {
    auto meta_it = point_meta_.find(kv.first);
    if (meta_it == point_meta_.end()) {
      stats.point_skipped++;
      continue;
    }
    PointMeta& meta = meta_it->second;
    if (!meta.ekf_init) {
      meta.ekf_init = true;
      meta.cov = Eigen::Matrix3d::Identity() * config_.vio_schur_point_cov_init;
    }

    Eigen::Vector3d tmp_res = kv.second.gv;
    for (const auto& factor : kv.second.W_by_clone) {
      tmp_res -= factor.second.transpose() * clone_dx.segment<6>(factor.first * 6);
    }

    Eigen::Matrix3d Rcov =
        kv.second.V * config_.vio_obs_std_dev * config_.vio_obs_std_dev;
    Rcov.diagonal().array() += 1e-9;
    Eigen::Matrix3d Spt = kv.second.V * meta.cov * kv.second.V.transpose() + Rcov;
    Spt = (Spt + Spt.transpose()) * 0.5;
    Spt.diagonal().array() += 1e-9;

    Eigen::Matrix3d Kpt_T = Spt.ldlt().solve(kv.second.V * meta.cov);
    Eigen::Matrix3d Kpt = Kpt_T.transpose();
    Eigen::Vector3d delta_p = Kpt * tmp_res;
    if (!delta_p.array().isFinite().all()) {
      stats.point_skipped++;
      continue;
    }

    Eigen::Vector3d p_w_new = kv.second.p_w + delta_p;
    if (p_w_new.norm() > 1e4) {
      stats.point_skipped++;
      continue;
    }

    meta.cov = (Eigen::Matrix3d::Identity() - Kpt * kv.second.V) * meta.cov;
    meta.cov = (meta.cov + meta.cov.transpose()) * 0.5;
    if (writeWorldPointToLandmark(kv.first, p_w_new)) {
      stats.point_updates++;
    } else {
      stats.point_skipped++;
    }
  }

  return stats;
}

void SchurVinsEkfEstimator::applyCurrentInc(const Vec15& dx) {
  current_state_.T_w_i.so3() = Sophus::SO3d::exp(dx.segment<3>(0)) *
                               current_state_.T_w_i.so3();
  current_state_.T_w_i.translation() += dx.segment<3>(3);
  current_state_.vel_w_i += dx.segment<3>(6);
  current_state_.ba += dx.segment<3>(9);
  current_state_.bg += dx.segment<3>(12);
}

void SchurVinsEkfEstimator::applyCloneInc(ClonePose& clone, const Vec6& dx) {
  clone.T_w_i.so3() = Sophus::SO3d::exp(dx.segment<3>(0)) * clone.T_w_i.so3();
  clone.T_w_i.translation() += dx.segment<3>(3);
}

bool SchurVinsEkfEstimator::worldPointFromLandmark(
    const Keypoint<double>& kpt, Eigen::Vector3d& p_w) const {
  auto host_it = clone_poses_.find(kpt.host_kf_id.frame_id);
  if (host_it == clone_poses_.end()) return false;
  if (kpt.host_kf_id.cam_id >= calib_.T_i_c.size()) return false;

  Eigen::Vector4d p_cam = StereographicParam<double>::unproject(kpt.direction);
  p_cam[3] = kpt.inv_dist;
  if (std::abs(p_cam[3]) < kMinDepth) return false;

  const Eigen::Vector4d p_host =
      (host_it->second.T_w_i * calib_.T_i_c[kpt.host_kf_id.cam_id]).matrix() *
      p_cam;
  if (std::abs(p_host[3]) < kMinDepth) return false;
  p_w = p_host.head<3>() / p_host[3];
  return p_w.array().isFinite().all();
}

bool SchurVinsEkfEstimator::writeWorldPointToLandmark(
    KeypointId lm_id, const Eigen::Vector3d& p_w) {
  if (!lmdb_.landmarkExists(lm_id)) return false;
  Keypoint<double>& kpt = lmdb_.getLandmark(lm_id);
  auto host_it = clone_poses_.find(kpt.host_kf_id.frame_id);
  if (host_it == clone_poses_.end()) return false;
  if (kpt.host_kf_id.cam_id >= calib_.T_i_c.size()) return false;

  const Sophus::SE3d T_w_c = host_it->second.T_w_i * calib_.T_i_c[kpt.host_kf_id.cam_id];
  const Eigen::Vector3d p_c = T_w_c.inverse() * p_w;
  const double dist = p_c.norm();
  if (!p_c.array().isFinite().all() || dist <= kMinDepth || p_c.z() <= kMinDepth) {
    return false;
  }

  Eigen::Vector4d p_h;
  p_h.head<3>() = p_c / dist;
  p_h[3] = 1.0 / dist;
  kpt.direction = StereographicParam<double>::project(p_h);
  kpt.inv_dist = p_h[3];
  return true;
}

int SchurVinsEkfEstimator::removeOutliers(
    const OpticalFlowResult::Ptr& opt_flow_meas) {
  int removed = 0;
  for (size_t cam_id = 0; cam_id < opt_flow_meas->observations.size(); cam_id++) {
    const TimeCamId tcid(opt_flow_meas->t_ns, cam_id);
    for (const auto& obs : opt_flow_meas->observations[cam_id]) {
      if (!lmdb_.landmarkExists(obs.first)) continue;
      Eigen::Vector3d p_w;
      if (!worldPointFromLandmark(lmdb_.getLandmark(obs.first), p_w)) continue;
      const double error =
          reprojectionError(p_w, tcid, obs.second.translation().cast<double>());
      if (error > config_.vio_schur_outlier_threshold) {
        lmdb_.removeObservations(obs.first, {tcid});
        removed++;
      }
    }
  }
  return removed;
}

int SchurVinsEkfEstimator::removePointOutliers() {
  std::vector<KeypointId> to_remove;
  for (const auto& kv : lmdb_.getLandmarks()) {
    Eigen::Vector3d p_w;
    if (!worldPointFromLandmark(kv.second, p_w)) {
      to_remove.push_back(kv.first);
      continue;
    }

    double total_error = 0.0;
    int num_obs = 0;
    for (const auto& obs : kv.second.obs) {
      if (clone_poses_.count(obs.first.frame_id) == 0) continue;
      const double error = reprojectionError(p_w, obs.first, obs.second);
      if (std::isfinite(error)) {
        total_error += error;
        num_obs++;
      }
    }

    if (num_obs > 0 &&
        total_error / num_obs > config_.vio_schur_point_outlier_threshold) {
      to_remove.push_back(kv.first);
    }
  }

  for (KeypointId lm_id : to_remove) {
    lmdb_.removeLandmark(lm_id);
    point_meta_.erase(lm_id);
  }
  return static_cast<int>(to_remove.size());
}

double SchurVinsEkfEstimator::reprojectionError(
    const Eigen::Vector3d& p_w, const TimeCamId& tcid,
    const Eigen::Vector2d& obs_px) const {
  Eigen::Vector2d residual;
  if (!computeObservation(p_w, tcid, obs_px, residual, nullptr, nullptr)) {
    return std::numeric_limits<double>::infinity();
  }
  return residual.norm();
}

SchurVinsEkfEstimator::WindowManagementStats SchurVinsEkfEstimator::manageWindow(
    const std::map<int64_t, int>& num_points_connected) {
  WindowManagementStats stats;
  auto regular_count = [&] {
    size_t count = 0;
    for (const auto& kv : clone_poses_) {
      if (kf_ids_.count(kv.first) == 0) count++;
    }
    return count;
  };

  while (regular_count() > max_regular_frames_) {
    int64_t id_to_remove = -1;
    for (const auto& kv : clone_poses_) {
      if (kf_ids_.count(kv.first) == 0 && kv.first != current_state_.t_ns) {
        id_to_remove = kv.first;
        break;
      }
    }
    if (id_to_remove < 0) break;
    removeClone(id_to_remove, false);
    stats.removed_regular_clones++;
  }

  while (kf_ids_.size() > max_kfs_) {
    const int64_t id_to_remove = selectKeyframeToMarginalize(num_points_connected);
    if (id_to_remove < 0) break;
    removeClone(id_to_remove, true);
    stats.removed_keyframes++;
  }

  return stats;
}

void SchurVinsEkfEstimator::removeClone(int64_t t_ns,
                                        bool remove_host_landmarks) {
  auto clone_it = clone_poses_.find(t_ns);
  if (clone_it == clone_poses_.end()) return;

  const int idx = clone_it->second.index;

  std::vector<KeypointId> landmarks_to_remove;
  std::vector<std::pair<KeypointId, std::set<TimeCamId>>> observations_to_remove;
  for (const auto& kv : lmdb_.getLandmarks()) {
    if (remove_host_landmarks && kv.second.host_kf_id.frame_id == t_ns) {
      landmarks_to_remove.push_back(kv.first);
    } else {
      std::set<TimeCamId> obs_to_remove;
      for (const auto& obs : kv.second.obs) {
        if (obs.first.frame_id == t_ns) obs_to_remove.insert(obs.first);
      }
      if (!obs_to_remove.empty()) observations_to_remove.emplace_back(kv.first, obs_to_remove);
    }
  }

  for (const auto& kv : observations_to_remove) lmdb_.removeObservations(kv.first, kv.second);
  for (KeypointId lm_id : landmarks_to_remove) {
    lmdb_.removeLandmark(lm_id);
    point_meta_.erase(lm_id);
  }

  removeCloneCovariance(idx);
  clone_poses_.erase(clone_it);
  kf_ids_.erase(t_ns);
  num_points_kf_.erase(t_ns);
  prev_opt_flow_res_.erase(t_ns);
  updateCloneIndices();
}

void SchurVinsEkfEstimator::removeCloneCovariance(int clone_index) {
  const int old_size = cov_.rows();
  const int remove_start = 15 + clone_index * 6;
  std::vector<int> keep;
  keep.reserve(old_size - 6);
  for (int i = 0; i < old_size; i++) {
    if (i < remove_start || i >= remove_start + 6) keep.push_back(i);
  }

  Eigen::MatrixXd new_cov(keep.size(), keep.size());
  for (size_t r = 0; r < keep.size(); r++) {
    for (size_t c = 0; c < keep.size(); c++) {
      new_cov(r, c) = cov_(keep[r], keep[c]);
    }
  }
  cov_ = (new_cov + new_cov.transpose()) * 0.5;
}

int64_t SchurVinsEkfEstimator::selectKeyframeToMarginalize(
    const std::map<int64_t, int>& num_points_connected) const {
  if (kf_ids_.empty()) return -1;
  if (kf_ids_.size() <= 2) return *kf_ids_.begin();

  const auto end_minus_2 = std::prev(kf_ids_.end(), 2);
  for (auto it = kf_ids_.begin(); it != end_minus_2; ++it) {
    const auto num_kf_it = num_points_kf_.find(*it);
    const int num_kf_points = num_kf_it == num_points_kf_.end() ? 0 : num_kf_it->second;
    const auto connected_it = num_points_connected.find(*it);
    const int connected = connected_it == num_points_connected.end() ? 0 : connected_it->second;
    if (num_kf_points <= 0 ||
        connected / static_cast<double>(num_kf_points) < config_.vio_kf_marg_feature_ratio) {
      return *it;
    }
  }

  const int64_t latest_kf = *kf_ids_.rbegin();
  double min_score = std::numeric_limits<double>::max();
  int64_t min_score_id = *kf_ids_.begin();

  for (auto it1 = kf_ids_.begin(); it1 != end_minus_2; ++it1) {
    const auto pose_it1 = clone_poses_.find(*it1);
    if (pose_it1 == clone_poses_.end()) continue;

    double denom = 0.0;
    for (auto it2 = kf_ids_.begin(); it2 != end_minus_2; ++it2) {
      const auto pose_it2 = clone_poses_.find(*it2);
      if (pose_it2 == clone_poses_.end()) continue;
      denom += 1.0 / ((pose_it1->second.T_w_i.translation() -
                       pose_it2->second.T_w_i.translation()).norm() +
                      1e-5);
    }

    const double dist_latest =
        (pose_it1->second.T_w_i.translation() -
         clone_poses_.at(latest_kf).T_w_i.translation()).norm();
    const double score = std::sqrt(dist_latest) * denom;
    if (score < min_score) {
      min_score = score;
      min_score_id = *it1;
    }
  }

  return min_score_id;
}

void SchurVinsEkfEstimator::publishState() {
  if (!out_state_queue) return;

  auto data = std::make_shared<PoseVelBiasState<double>>(
      current_state_.t_ns, current_state_.T_w_i, current_state_.vel_w_i,
      current_state_.bg, current_state_.ba);
  out_state_queue->push(data);
}

void SchurVinsEkfEstimator::publishVisualization(
    const OpticalFlowResult::Ptr& opt_flow_meas) {
  if (!out_vis_queue) return;

  VioVisualizationData::Ptr data(new VioVisualizationData);
  data->t_ns = current_state_.t_ns;
  data->states.emplace_back(current_state_.T_w_i);
  for (const auto& kv : clone_poses_) data->frames.emplace_back(kv.second.T_w_i);

  for (const auto& kv : lmdb_.getLandmarks()) {
    Eigen::Vector3d p_w;
    if (worldPointFromLandmark(kv.second, p_w)) {
      data->points.emplace_back(p_w);
      data->point_ids.push_back(static_cast<int>(kv.first));
    }
  }

  data->projections.resize(opt_flow_meas->observations.size());
  data->opt_flow_res = opt_flow_meas;
  out_vis_queue->push(data);
}

Eigen::Vector3d SchurVinsEkfEstimator::getBearing(
    const TimeCamId& tcid, const Eigen::Vector2d& obs_px) const {
  Eigen::Vector4d bearing4;
  if (tcid.cam_id >= calib_.intrinsics.size() ||
      !calib_.intrinsics[tcid.cam_id].unproject(obs_px, bearing4)) {
    return Vec3::Constant(std::numeric_limits<double>::quiet_NaN());
  }
  return bearing4.head<3>();
}

double SchurVinsEkfEstimator::focalLength() const {
  if (config_.vio_schur_focal_length > 0.0) return config_.vio_schur_focal_length;
  if (calib_.intrinsics.empty()) return 1.0;

  const Eigen::VectorXd params = calib_.intrinsics[0].getParam();
  if (params.size() >= 2) return 0.5 * (std::abs(params[0]) + std::abs(params[1]));
  if (params.size() == 1) return std::abs(params[0]);
  return 1.0;
}

Eigen::Matrix3d SchurVinsEkfEstimator::skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d mat;
  mat << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return mat;
}

}  // namespace basalt
