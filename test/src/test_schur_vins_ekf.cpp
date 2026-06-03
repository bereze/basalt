#include <basalt/camera/kannala_brandt_camera4.hpp>
#include <basalt/camera/stereographic_param.hpp>
#include <basalt/vi_estimator/schur_vins_ekf.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>

#include "gtest/gtest.h"

namespace basalt {

class SchurVinsEkfEstimatorTestAccessor {
 public:
  static void initialize(SchurVinsEkfEstimator& estimator, int64_t t_ns,
                         const Sophus::SE3d& T_w_i) {
    estimator.initializeState(t_ns, T_w_i, Eigen::Vector3d::Zero(),
                              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  }

  static void augmentWithPose(SchurVinsEkfEstimator& estimator, int64_t t_ns,
                              const Sophus::SE3d& T_w_i) {
    estimator.current_state_.t_ns = t_ns;
    estimator.current_state_.T_w_i = T_w_i;
    estimator.augmentClone(t_ns);
  }

  static void markKeyframe(SchurVinsEkfEstimator& estimator, int64_t t_ns,
                           int num_points = 0) {
    estimator.kf_ids_.emplace(t_ns);
    estimator.clone_poses_.at(t_ns).is_keyframe = true;
    estimator.num_points_kf_[t_ns] = num_points;
  }

  static void setMaxStates(SchurVinsEkfEstimator& estimator, size_t count) {
    estimator.setMaxStates(count);
  }

  static void setMaxKfs(SchurVinsEkfEstimator& estimator, size_t count) {
    estimator.setMaxKfs(count);
  }

  static size_t cloneCount(const SchurVinsEkfEstimator& estimator) {
    return estimator.clone_poses_.size();
  }

  static size_t keyframeCount(const SchurVinsEkfEstimator& estimator) {
    return estimator.kf_ids_.size();
  }

  static size_t regularFrameCount(const SchurVinsEkfEstimator& estimator) {
    size_t count = 0;
    for (const auto& kv : estimator.clone_poses_) {
      if (estimator.kf_ids_.count(kv.first) == 0) count++;
    }
    return count;
  }

  static bool hasClone(const SchurVinsEkfEstimator& estimator, int64_t t_ns) {
    return estimator.clone_poses_.count(t_ns) > 0;
  }

  static int cloneIndex(const SchurVinsEkfEstimator& estimator, int64_t t_ns) {
    return estimator.cloneIndex(t_ns);
  }

  static int covRows(const SchurVinsEkfEstimator& estimator) {
    return static_cast<int>(estimator.cov_.rows());
  }

  static bool covFinite(const SchurVinsEkfEstimator& estimator) {
    return estimator.cov_.array().isFinite().all();
  }

  static double covSymmetryError(const SchurVinsEkfEstimator& estimator) {
    return (estimator.cov_ - estimator.cov_.transpose()).norm();
  }

  static double covTrace(const SchurVinsEkfEstimator& estimator) {
    return estimator.cov_.trace();
  }

  static double minCovEigenvalue(const SchurVinsEkfEstimator& estimator) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(estimator.cov_);
    return es.eigenvalues().minCoeff();
  }

  static Sophus::SE3d clonePose(const SchurVinsEkfEstimator& estimator,
                                int64_t t_ns) {
    return estimator.clone_poses_.at(t_ns).T_w_i;
  }

  static Sophus::SE3d currentPose(const SchurVinsEkfEstimator& estimator) {
    return estimator.current_state_.T_w_i;
  }

  static Eigen::Vector3d velocity(const SchurVinsEkfEstimator& estimator) {
    return estimator.current_state_.vel_w_i;
  }

  static Eigen::Vector3d accelBias(const SchurVinsEkfEstimator& estimator) {
    return estimator.current_state_.ba;
  }

  static Eigen::Vector3d gyroBias(const SchurVinsEkfEstimator& estimator) {
    return estimator.current_state_.bg;
  }

  static void applyCurrentInc(SchurVinsEkfEstimator& estimator,
                              const Eigen::Matrix<double, 15, 1>& dx) {
    estimator.applyCurrentInc(dx);
  }

  static void manageWindow(SchurVinsEkfEstimator& estimator) {
    estimator.manageWindow(std::map<int64_t, int>());
  }

  static void removeClone(SchurVinsEkfEstimator& estimator, int64_t t_ns,
                          bool remove_host_landmarks) {
    estimator.removeClone(t_ns, remove_host_landmarks);
  }

  static void addLandmark(SchurVinsEkfEstimator& estimator, KeypointId lm_id,
                          int64_t host_t_ns, const Eigen::Vector3d& p_w) {
    const Sophus::SE3d T_c_w = estimator.calib_.T_i_c[0].inverse() *
                               estimator.clone_poses_.at(host_t_ns).T_w_i.inverse();
    const Eigen::Vector3d p_c = T_c_w * p_w;

    Keypoint<double> kpt;
    kpt.host_kf_id = TimeCamId(host_t_ns, 0);
    kpt.direction = StereographicParam<double>::project(p_c.homogeneous());
    kpt.inv_dist = 1.0 / p_c.norm();
    estimator.lmdb_.addLandmark(lm_id, kpt);
    estimator.point_meta_[lm_id].local_status = true;
  }

  static void addObservation(SchurVinsEkfEstimator& estimator, KeypointId lm_id,
                             const TimeCamId& tcid,
                             const Eigen::Vector2d& px) {
    KeypointObservation<double> obs;
    obs.kpt_id = static_cast<int>(lm_id);
    obs.pos = px;
    estimator.lmdb_.addObservation(tcid, obs);
  }

  static bool hasLandmark(const SchurVinsEkfEstimator& estimator,
                          KeypointId lm_id) {
    return estimator.lmdb_.landmarkExists(lm_id);
  }

  static bool pointInitialized(const SchurVinsEkfEstimator& estimator,
                               KeypointId lm_id) {
    const auto it = estimator.point_meta_.find(lm_id);
    return it != estimator.point_meta_.end() && it->second.ekf_init;
  }

  static double minPointCovEigenvalue(const SchurVinsEkfEstimator& estimator,
                                      KeypointId lm_id) {
    const auto it = estimator.point_meta_.find(lm_id);
    if (it == estimator.point_meta_.end()) {
      return -std::numeric_limits<double>::infinity();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(it->second.cov);
    return es.eigenvalues().minCoeff();
  }

  static void visualUpdate(SchurVinsEkfEstimator& estimator) {
    estimator.visualUpdate();
  }
};

}  // namespace basalt

namespace {

basalt::Calibration<double> makeStereoCalibration() {
  basalt::Calibration<double> calib;
  calib.T_i_c.emplace_back(Sophus::SE3d());

  Sophus::SE3d T_i_c1;
  T_i_c1.translation() = Eigen::Vector3d(0.11, 0.0, 0.0);
  calib.T_i_c.emplace_back(T_i_c1);

  basalt::GenericCamera<double> cam;
  cam.variant = basalt::KannalaBrandtCamera4<double>::getTestProjections()[0];
  calib.intrinsics.emplace_back(cam);
  calib.intrinsics.emplace_back(cam);
  calib.resolution.emplace_back(640, 480);
  calib.resolution.emplace_back(640, 480);
  return calib;
}

basalt::VioConfig makeSchurConfig() {
  basalt::VioConfig config;
  config.vio_backend_type = basalt::VioBackendType::SCHUR_EKF;
  config.vio_schur_max_frames = 2;
  config.vio_max_kfs = 2;
  config.vio_schur_min_obs = 3;
  config.vio_obs_std_dev = 1.0;
  config.vio_schur_huber_thresh = 5.0;
  config.vio_min_triangulation_dist = 0.01;
  config.vio_schur_max_iterations = 2;
  config.vio_schur_use_pixel_residual = true;
  return config;
}

Sophus::SE3d makePose(double x) {
  Sophus::SE3d T_w_i;
  T_w_i.translation() = Eigen::Vector3d(x, 0.0, 0.0);
  return T_w_i;
}

bool projectPoint(const basalt::Calibration<double>& calib,
                  const Sophus::SE3d& T_w_i, size_t cam_id,
                  const Eigen::Vector3d& p_w, Eigen::Vector2d& px) {
  const Sophus::SE3d T_c_w = calib.T_i_c[cam_id].inverse() * T_w_i.inverse();
  return calib.intrinsics[cam_id].project(T_c_w * p_w, px);
}

}  // namespace

TEST(SchurEkfTest, ConfigRoundTripKeepsBackendAndParameters) {
  basalt::VioConfig defaults;
  EXPECT_EQ(basalt::VioBackendType::SQRT_BA, defaults.vio_backend_type);

  basalt::VioConfig config;
  config.vio_backend_type = basalt::VioBackendType::SCHUR_EKF;
  config.vio_schur_max_frames = 7;
  config.vio_schur_huber_thresh = 2.25;
  config.vio_schur_min_obs = 5;
  config.vio_schur_outlier_threshold = 6.5;
  config.vio_schur_point_outlier_threshold = 4.5;
  config.vio_schur_point_cov_init = 42.0;
  config.vio_schur_focal_length = 320.0;
  config.vio_schur_max_iterations = 3;
  config.vio_schur_use_pixel_residual = true;

  const std::string path = "/tmp/basalt_schur_ekf_config_roundtrip.json";
  config.save(path);

  basalt::VioConfig loaded;
  loaded.load(path);

  EXPECT_EQ(basalt::VioBackendType::SCHUR_EKF, loaded.vio_backend_type);
  EXPECT_EQ(7, loaded.vio_schur_max_frames);
  EXPECT_DOUBLE_EQ(2.25, loaded.vio_schur_huber_thresh);
  EXPECT_EQ(5, loaded.vio_schur_min_obs);
  EXPECT_DOUBLE_EQ(6.5, loaded.vio_schur_outlier_threshold);
  EXPECT_DOUBLE_EQ(4.5, loaded.vio_schur_point_outlier_threshold);
  EXPECT_DOUBLE_EQ(42.0, loaded.vio_schur_point_cov_init);
  EXPECT_DOUBLE_EQ(320.0, loaded.vio_schur_focal_length);
  EXPECT_EQ(3, loaded.vio_schur_max_iterations);
  EXPECT_TRUE(loaded.vio_schur_use_pixel_residual);
}

TEST(SchurEkfTest, FactoryCreatesSchurBackendWhenRequested) {
  basalt::VioConfig config = makeSchurConfig();
  basalt::Calibration<double> calib = makeStereoCalibration();

#if defined(BASALT_INSTANTIATIONS_FLOAT)
  basalt::VioEstimatorBase::Ptr vio = basalt::VioEstimatorFactory::getVioEstimator(
      config, calib, Eigen::Vector3d(0.0, 0.0, -9.81), true, false);
#elif defined(BASALT_INSTANTIATIONS_DOUBLE)
  basalt::VioEstimatorBase::Ptr vio = basalt::VioEstimatorFactory::getVioEstimator(
      config, calib, Eigen::Vector3d(0.0, 0.0, -9.81), true, true);
#else
  GTEST_SKIP() << "Basalt VIO estimator instantiations are disabled";
#endif

  ASSERT_NE(nullptr, vio);
  EXPECT_NE(nullptr, dynamic_cast<basalt::SchurVinsEkfEstimator*>(vio.get()));
}

TEST(SchurEkfTest, DestructorStopsEmptyProcessingThread) {
  basalt::VioConfig config = makeSchurConfig();
  basalt::SchurVinsEkfEstimator estimator(Eigen::Vector3d(0.0, 0.0, -9.81),
                                          makeStereoCalibration(), config);
  estimator.initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
}

TEST(SchurEkfTest, CurrentStateIncrementUsesExpectedOrder) {
  basalt::SchurVinsEkfEstimator estimator(Eigen::Vector3d(0.0, 0.0, -9.81),
                                          makeStereoCalibration(), makeSchurConfig());
  basalt::SchurVinsEkfEstimatorTestAccessor::initialize(estimator, 0, Sophus::SE3d());

  Eigen::Matrix<double, 15, 1> dx = Eigen::Matrix<double, 15, 1>::Zero();
  dx.segment<3>(0) = Eigen::Vector3d(0.01, -0.02, 0.03);
  dx.segment<3>(3) = Eigen::Vector3d(1.0, 2.0, 3.0);
  dx.segment<3>(6) = Eigen::Vector3d(0.4, 0.5, 0.6);
  dx.segment<3>(9) = Eigen::Vector3d(-0.01, 0.02, -0.03);
  dx.segment<3>(12) = Eigen::Vector3d(0.07, -0.08, 0.09);

  basalt::SchurVinsEkfEstimatorTestAccessor::applyCurrentInc(estimator, dx);

  const Sophus::SE3d pose =
      basalt::SchurVinsEkfEstimatorTestAccessor::currentPose(estimator);
  EXPECT_NEAR(0.0, (pose.so3().log() - dx.segment<3>(0)).norm(), 1e-12);
  EXPECT_NEAR(0.0, (pose.translation() - dx.segment<3>(3)).norm(), 1e-12);
  EXPECT_NEAR(0.0, (basalt::SchurVinsEkfEstimatorTestAccessor::velocity(estimator) -
                    dx.segment<3>(6)).norm(), 1e-12);
  EXPECT_NEAR(0.0, (basalt::SchurVinsEkfEstimatorTestAccessor::accelBias(estimator) -
                    dx.segment<3>(9)).norm(), 1e-12);
  EXPECT_NEAR(0.0, (basalt::SchurVinsEkfEstimatorTestAccessor::gyroBias(estimator) -
                    dx.segment<3>(12)).norm(), 1e-12);
}

TEST(SchurEkfTest, CloneWindowKeepsRegularAndKeyframesInOneMap) {
  basalt::Calibration<double> calib = makeStereoCalibration();
  basalt::SchurVinsEkfEstimator estimator(Eigen::Vector3d(0.0, 0.0, -9.81),
                                          calib, makeSchurConfig());
  basalt::SchurVinsEkfEstimatorTestAccessor::initialize(estimator, 0, Sophus::SE3d());

  const Eigen::Vector3d p_w(0.2, 0.1, 4.0);
  basalt::SchurVinsEkfEstimatorTestAccessor::augmentWithPose(estimator, 10, makePose(0.0));
  basalt::SchurVinsEkfEstimatorTestAccessor::markKeyframe(estimator, 10, 1);
  basalt::SchurVinsEkfEstimatorTestAccessor::addLandmark(estimator, 100, 10, p_w);
  basalt::SchurVinsEkfEstimatorTestAccessor::augmentWithPose(estimator, 20, makePose(0.1));
  basalt::SchurVinsEkfEstimatorTestAccessor::augmentWithPose(estimator, 30, makePose(0.2));

  for (int64_t frame_id : {int64_t(10), int64_t(20), int64_t(30)}) {
    Eigen::Vector2d px;
    ASSERT_TRUE(projectPoint(
        calib, basalt::SchurVinsEkfEstimatorTestAccessor::clonePose(estimator, frame_id),
        0, p_w, px));
    basalt::SchurVinsEkfEstimatorTestAccessor::addObservation(
        estimator, 100, basalt::TimeCamId(frame_id, 0), px);
  }

  EXPECT_EQ(3u, basalt::SchurVinsEkfEstimatorTestAccessor::cloneCount(estimator));
  EXPECT_EQ(1u, basalt::SchurVinsEkfEstimatorTestAccessor::keyframeCount(estimator));
  EXPECT_EQ(2u, basalt::SchurVinsEkfEstimatorTestAccessor::regularFrameCount(estimator));
  EXPECT_EQ(33, basalt::SchurVinsEkfEstimatorTestAccessor::covRows(estimator));

  basalt::SchurVinsEkfEstimatorTestAccessor::setMaxStates(estimator, 1);
  basalt::SchurVinsEkfEstimatorTestAccessor::manageWindow(estimator);

  EXPECT_TRUE(basalt::SchurVinsEkfEstimatorTestAccessor::hasClone(estimator, 10));
  EXPECT_FALSE(basalt::SchurVinsEkfEstimatorTestAccessor::hasClone(estimator, 20));
  EXPECT_TRUE(basalt::SchurVinsEkfEstimatorTestAccessor::hasClone(estimator, 30));
  EXPECT_EQ(2u, basalt::SchurVinsEkfEstimatorTestAccessor::cloneCount(estimator));
  EXPECT_EQ(1u, basalt::SchurVinsEkfEstimatorTestAccessor::keyframeCount(estimator));
  EXPECT_EQ(1u, basalt::SchurVinsEkfEstimatorTestAccessor::regularFrameCount(estimator));
  EXPECT_EQ(27, basalt::SchurVinsEkfEstimatorTestAccessor::covRows(estimator));
  EXPECT_TRUE(basalt::SchurVinsEkfEstimatorTestAccessor::hasLandmark(estimator, 100));

  basalt::SchurVinsEkfEstimatorTestAccessor::removeClone(estimator, 10, true);

  EXPECT_FALSE(basalt::SchurVinsEkfEstimatorTestAccessor::hasClone(estimator, 10));
  EXPECT_FALSE(basalt::SchurVinsEkfEstimatorTestAccessor::hasLandmark(estimator, 100));
  EXPECT_EQ(1u, basalt::SchurVinsEkfEstimatorTestAccessor::cloneCount(estimator));
  EXPECT_EQ(0u, basalt::SchurVinsEkfEstimatorTestAccessor::keyframeCount(estimator));
  EXPECT_EQ(21, basalt::SchurVinsEkfEstimatorTestAccessor::covRows(estimator));
  EXPECT_EQ(0, basalt::SchurVinsEkfEstimatorTestAccessor::cloneIndex(estimator, 30));
  EXPECT_TRUE(basalt::SchurVinsEkfEstimatorTestAccessor::covFinite(estimator));
  EXPECT_NEAR(0.0, basalt::SchurVinsEkfEstimatorTestAccessor::covSymmetryError(estimator),
              1e-12);
}

TEST(SchurEkfTest, SyntheticVisualUpdateKeepsStateAndCovarianceFinite) {
  basalt::Calibration<double> calib = makeStereoCalibration();
  basalt::SchurVinsEkfEstimator estimator(Eigen::Vector3d(0.0, 0.0, -9.81),
                                          calib, makeSchurConfig());
  basalt::SchurVinsEkfEstimatorTestAccessor::initialize(estimator, 0, Sophus::SE3d());

  const std::array<int64_t, 3> frame_ids = {100, 200, 300};
  for (size_t i = 0; i < frame_ids.size(); i++) {
    basalt::SchurVinsEkfEstimatorTestAccessor::augmentWithPose(
        estimator, frame_ids[i], makePose(0.08 * static_cast<double>(i)));
  }
  basalt::SchurVinsEkfEstimatorTestAccessor::markKeyframe(estimator, frame_ids[0], 1);

  const basalt::KeypointId lm_id = 7;
  const Eigen::Vector3d p_w(0.2, 0.1, 4.0);
  basalt::SchurVinsEkfEstimatorTestAccessor::addLandmark(estimator, lm_id,
                                                         frame_ids[0], p_w);

  for (int64_t frame_id : frame_ids) {
    const Sophus::SE3d T_w_i =
        basalt::SchurVinsEkfEstimatorTestAccessor::clonePose(estimator, frame_id);
    for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
      Eigen::Vector2d px;
      ASSERT_TRUE(projectPoint(calib, T_w_i, cam_id, p_w, px));
      basalt::SchurVinsEkfEstimatorTestAccessor::addObservation(
          estimator, lm_id, basalt::TimeCamId(frame_id, cam_id), px);
    }
  }

  ASSERT_EQ(33, basalt::SchurVinsEkfEstimatorTestAccessor::covRows(estimator));
  const double trace_before =
      basalt::SchurVinsEkfEstimatorTestAccessor::covTrace(estimator);

  basalt::SchurVinsEkfEstimatorTestAccessor::visualUpdate(estimator);

  EXPECT_EQ(33, basalt::SchurVinsEkfEstimatorTestAccessor::covRows(estimator));
  EXPECT_TRUE(basalt::SchurVinsEkfEstimatorTestAccessor::covFinite(estimator));
  EXPECT_NEAR(0.0, basalt::SchurVinsEkfEstimatorTestAccessor::covSymmetryError(estimator),
              1e-9);
  EXPECT_GE(basalt::SchurVinsEkfEstimatorTestAccessor::minCovEigenvalue(estimator),
            -1e-10);
  EXPECT_TRUE(basalt::SchurVinsEkfEstimatorTestAccessor::pointInitialized(estimator,
                                                                          lm_id));
  EXPECT_GE(basalt::SchurVinsEkfEstimatorTestAccessor::minPointCovEigenvalue(estimator,
                                                                             lm_id),
            -1e-10);
  EXPECT_LE(basalt::SchurVinsEkfEstimatorTestAccessor::covTrace(estimator),
            trace_before + 1e-9);
}
