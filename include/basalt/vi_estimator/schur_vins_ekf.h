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
#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include <basalt/utils/time_utils.hpp>
#include <basalt/vi_estimator/ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

namespace basalt {

class SchurVinsEkfEstimatorTestAccessor;

class SchurVinsEkfEstimator : public VioEstimatorBase {
 public:
  using Ptr = std::shared_ptr<SchurVinsEkfEstimator>;

  explicit SchurVinsEkfEstimator(const Eigen::Vector3d& g,
                                 const Calibration<double>& calib,
                                 const VioConfig& config);
  ~SchurVinsEkfEstimator();

  void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                  const Eigen::Vector3d& vel_w_i, const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;
  void initialize(const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;

  void maybe_join() override;

  Sophus::SE3d getT_w_i_init() override { return T_w_i_init_; }

  void setMaxStates(size_t val) override;
  void setMaxKfs(size_t val) override;

  void addIMUToQueue(const ImuData<double>::Ptr& data) override;
  void addVisionToQueue(const OpticalFlowResult::Ptr& data) override;

  void debug_finalize() override;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  using Vec2 = Eigen::Vector2d;
  using Vec3 = Eigen::Vector3d;
  using Vec6 = Eigen::Matrix<double, 6, 1>;
  using Vec12 = Eigen::Matrix<double, 12, 1>;
  using Vec15 = Eigen::Matrix<double, 15, 1>;
  using Mat23 = Eigen::Matrix<double, 2, 3>;
  using Mat26 = Eigen::Matrix<double, 2, 6>;
  using Mat63 = Eigen::Matrix<double, 6, 3>;
  using Mat6 = Eigen::Matrix<double, 6, 6>;
  using Mat12 = Eigen::Matrix<double, 12, 12>;
  using Mat15 = Eigen::Matrix<double, 15, 15>;

  friend class SchurVinsEkfEstimatorTestAccessor;

  struct CurrentState {
    int64_t t_ns = 0;
    Sophus::SE3d T_w_i;
    Vec3 vel_w_i = Vec3::Zero();
    Vec3 bg = Vec3::Zero();
    Vec3 ba = Vec3::Zero();
  };

  struct ClonePose {
    Sophus::SE3d T_w_i;
    int index = -1;
    bool is_keyframe = false;
  };

  struct PointMeta {
    bool ekf_init = false;
    bool local_status = true;
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
  };

  struct PointLinData {
    Eigen::Vector3d p_w = Eigen::Vector3d::Zero();
    Eigen::Matrix3d V = Eigen::Matrix3d::Zero();
    Eigen::Vector3d gv = Eigen::Vector3d::Zero();
    std::map<int, Mat63> W_by_clone;
  };

  struct VisualUpdateStats {
    int candidate_landmarks = 0;
    int schur_landmarks = 0;
    int valid_observations = 0;
    int iterations = 0;
    int point_updates = 0;
    int point_skipped = 0;
    double residual_sum = 0.0;
    double residual_max = 0.0;
    double current_dx_norm = 0.0;
    double clone_dx_norm = 0.0;
  };

  struct StateUpdateStats {
    bool updated = false;
    int point_updates = 0;
    int point_skipped = 0;
    double current_dx_norm = 0.0;
    double clone_dx_norm = 0.0;
  };

  struct WindowManagementStats {
    int removed_regular_clones = 0;
    int removed_keyframes = 0;
  };

  void processingLoop(Eigen::Vector3d bg, Eigen::Vector3d ba);
  ImuData<double>::Ptr popFromImuDataQueue();
  void calibrateImu(ImuData<double>& data) const;

  void initializeState(int64_t t_ns, const Sophus::SE3d& T_w_i,
                       const Vec3& vel_w_i, const Vec3& bg,
                       const Vec3& ba);
  void initCovariance();
  void initNoise();
  void propagate(double dt, const Vec3& accel_meas, const Vec3& gyro_meas);
  void predictState(double dt, const Vec3& accel, const Vec3& gyro);

  bool measure(const OpticalFlowResult::Ptr& opt_flow_meas);
  void augmentClone(int64_t t_ns);
  void updateCloneIndices();
  int cloneIndex(int64_t t_ns) const;

  void registerObservations(const OpticalFlowResult::Ptr& opt_flow_meas,
                            int& connected0,
                            std::unordered_set<KeypointId>& unconnected_obs0,
                            std::map<int64_t, int>& num_points_connected);
  int maybeCreateKeyframe(const OpticalFlowResult::Ptr& opt_flow_meas,
                          int connected0,
                          const std::unordered_set<KeypointId>& unconnected_obs0);
  bool triangulateLandmark(KeypointId lm_id,
                           const OpticalFlowResult::Ptr& opt_flow_meas);

  void visualUpdate();
  bool addLandmarkLinearization(KeypointId lm_id, const Keypoint<double>& kpt,
                                Eigen::MatrixXd& H, Eigen::VectorXd& b,
                                PointLinData& lin_data,
                                VisualUpdateStats& stats) const;
  bool computeObservation(const Eigen::Vector3d& p_w, const TimeCamId& tcid,
                          const Eigen::Vector2d& obs_px,
                          Eigen::Vector2d& residual, Mat26* J_clone,
                          Mat23* J_point) const;
  StateUpdateStats stateUpdate(
      const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
      const Eigen::aligned_unordered_map<KeypointId, PointLinData>& point_data,
      bool update_points);
  void applyCurrentInc(const Vec15& dx);
  void applyCloneInc(ClonePose& clone, const Vec6& dx);

  bool worldPointFromLandmark(const Keypoint<double>& kpt,
                              Eigen::Vector3d& p_w) const;
  bool writeWorldPointToLandmark(KeypointId lm_id, const Eigen::Vector3d& p_w);

  int removeOutliers(const OpticalFlowResult::Ptr& opt_flow_meas);
  int removePointOutliers();
  double reprojectionError(const Eigen::Vector3d& p_w, const TimeCamId& tcid,
                           const Eigen::Vector2d& obs_px) const;

  WindowManagementStats manageWindow(
      const std::map<int64_t, int>& num_points_connected);
  void removeClone(int64_t t_ns, bool remove_host_landmarks);
  void removeCloneCovariance(int clone_index);
  int64_t selectKeyframeToMarginalize(
      const std::map<int64_t, int>& num_points_connected) const;

  void publishState();
  void publishVisualization(const OpticalFlowResult::Ptr& opt_flow_meas);
  Eigen::Vector3d getBearing(const TimeCamId& tcid,
                             const Eigen::Vector2d& obs_px) const;
  double focalLength() const;

  static Eigen::Matrix3d skew(const Eigen::Vector3d& v);

  Calibration<double> calib_;
  VioConfig config_;
  Vec3 gravity_;

  CurrentState current_state_;
  Sophus::SE3d T_w_i_init_;
  bool initialized_ = false;
  bool opt_started_ = false;
  bool take_kf_ = true;
  int frames_after_kf_ = 0;

  Eigen::MatrixXd cov_;
  Mat12 imu_noise_ = Mat12::Zero();

  Eigen::aligned_map<int64_t, ClonePose> clone_poses_;
  std::set<int64_t> kf_ids_;
  std::map<int64_t, int> num_points_kf_;
  Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res_;

  LandmarkDatabase<double> lmdb_;
  Eigen::aligned_unordered_map<KeypointId, PointMeta> point_meta_;

  size_t max_regular_frames_;
  size_t max_kfs_;
  double focal_length_;

  std::shared_ptr<std::thread> processing_thread_;
  ExecutionStats stats_all_;
  ExecutionStats stats_sums_;
};

}  // namespace basalt
