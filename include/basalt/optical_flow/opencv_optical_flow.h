/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <thread>

#include <opencv2/video/tracking.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patterns.h>

#include <basalt/utils/keypoints.h>

namespace basalt {

/// Frame-to-frame optical flow that tracks keypoints with OpenCV's pyramidal
/// Lucas-Kanade tracker (cv::calcOpticalFlowPyrLK) instead of basalt's own
/// inverse-compositional patch tracker used by FrameToFrameOpticalFlow.
/// Feature detection, forward-backward outlier rejection and stereo epipolar
/// filtering follow the exact same scheme as FrameToFrameOpticalFlow, so the
/// only thing that differs between the two classes is the KLT tracking
/// implementation itself -- this makes them directly comparable.
template <typename Scalar, template <typename> typename Pattern>
class OpenCVOpticalFlow : public OpticalFlowBase {
 public:
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  OpenCVOpticalFlow(const VioConfig& config,
                    const basalt::Calibration<double>& calib)
      : t_ns(-1),
        frame_counter(0),
        last_keypoint_id(0),
        config(config),
        win_size(21, 21),
        term_criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                      config.optical_flow_max_iterations, 0.01) {
    input_queue.set_capacity(10);

    this->calib = calib.cast<Scalar>();

    patch_coord = Pattern<Scalar>::pattern2.template cast<float>();

    if (calib.intrinsics.size() > 1) {
      Eigen::Matrix4d Ed;
      Sophus::SE3d T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      computeEssential(T_i_j, Ed);
      E = Ed.cast<Scalar>();
    }

    processing_thread.reset(
        new std::thread(&OpenCVOpticalFlow::processingLoop, this));
  }

  ~OpenCVOpticalFlow() { processing_thread->join(); }

  void processingLoop() {
    OpticalFlowInput::Ptr input_ptr;

    while (true) {
      input_queue.pop(input_ptr);

      if (!input_ptr.get()) {
        if (output_queue) output_queue->push(nullptr);
        break;
      }

      Timer timer;
      processFrame(input_ptr->t_ns, input_ptr);
      recordFrameTime(timer.elapsed());
    }
  }

  // Basalt images are stored as 16-bit (8-bit value shifted left by 8), so we
  // shift back down to get a regular 8-bit image for OpenCV.
  static cv::Mat toCvMat8U(const basalt::Image<const uint16_t>& img_raw) {
    cv::Mat img(img_raw.h, img_raw.w, CV_8U);

    uint8_t* dst = img.ptr();
    const uint16_t* src = img_raw.ptr;

    for (size_t i = 0; i < img_raw.size(); i++) {
      dst[i] = src[i] >> 8;
    }

    return img;
  }

  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    std::vector<cv::Mat> new_images(calib.intrinsics.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                      [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                          const auto& img = *new_img_vec->img_data[i].img;
                          const basalt::Image<const uint16_t> img_raw(
                              img.ptr, img.w, img.h, img.pitch);
                          new_images[i] = toCvMat8U(img_raw);
                        }
                      });

    if (t_ns < 0) {
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      transforms->observations.resize(calib.intrinsics.size());
      transforms->t_ns = t_ns;

      images = new_images;

      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();

    } else {
      t_ns = curr_t_ns;

      old_images = images;
      images = new_images;

      OpticalFlowResult::Ptr new_transforms;
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->observations.resize(calib.intrinsics.size());
      new_transforms->t_ns = t_ns;

      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        trackPoints(old_images.at(i), images.at(i),
                    transforms->observations[i],
                    new_transforms->observations[i]);
      }

      transforms = new_transforms;
      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();
    }

    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      output_queue->push(transforms);
    }

    frame_counter++;
  }

  // Track points from img_1 to img_2 with OpenCV's pyramidal KLT tracker,
  // then track back from img_2 to img_1 and discard points whose recovered
  // position is too far from the original one (forward-backward check).
  void trackPoints(const cv::Mat& img_1, const cv::Mat& img_2,
                   const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_1,
                   Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_2) const {
    transform_map_2.clear();

    const size_t num_points = transform_map_1.size();
    if (num_points == 0) return;

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_transforms;
    std::vector<cv::Point2f> pts_1;

    ids.reserve(num_points);
    init_transforms.reserve(num_points);
    pts_1.reserve(num_points);

    for (const auto& kv : transform_map_1) {
      ids.push_back(kv.first);
      init_transforms.push_back(kv.second);
      pts_1.emplace_back(kv.second.translation().x(),
                         kv.second.translation().y());
    }

    std::vector<cv::Point2f> pts_2;
    std::vector<uchar> status_fwd;
    std::vector<float> err_fwd;

    cv::calcOpticalFlowPyrLK(img_1, img_2, pts_1, pts_2, status_fwd, err_fwd,
                             win_size, config.optical_flow_levels,
                             term_criteria);

    std::vector<cv::Point2f> pts_1_recovered;
    std::vector<uchar> status_bwd;
    std::vector<float> err_bwd;

    cv::calcOpticalFlowPyrLK(img_2, img_1, pts_2, pts_1_recovered, status_bwd,
                             err_bwd, win_size, config.optical_flow_levels,
                             term_criteria);

    const float filter_margin = 2;

    for (size_t i = 0; i < ids.size(); i++) {
      if (!status_fwd[i] || !status_bwd[i]) continue;

      if (pts_2[i].x < filter_margin ||
          pts_2[i].x >= img_2.cols - filter_margin ||
          pts_2[i].y < filter_margin ||
          pts_2[i].y >= img_2.rows - filter_margin)
        continue;

      const float dist2 = (Eigen::Vector2f(pts_1[i].x, pts_1[i].y) -
                           Eigen::Vector2f(pts_1_recovered[i].x,
                                           pts_1_recovered[i].y))
                              .squaredNorm();

      if (dist2 >= config.optical_flow_max_recovered_dist2) continue;

      Eigen::AffineCompact2f transform = init_transforms[i];
      transform.translation() = Eigen::Vector2f(pts_2[i].x, pts_2[i].y);

      transform_map_2[ids[i]] = transform;
    }
  }

  void addPoints() {
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    for (const auto& kv : transforms->observations.at(0)) {
      pts0.emplace_back(kv.second.translation().cast<double>());
    }

    KeypointsData kd;

    const auto& img0 = *transforms->input_images->img_data[0].img;
    const basalt::Image<const uint16_t> img_raw(img0.ptr, img0.w, img0.h,
                                                img0.pitch);

    detectKeypoints(img_raw, kd, config.optical_flow_detection_grid_size, 1,
                    pts0);

    Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
        new_poses1;

    for (size_t i = 0; i < kd.corners.size(); i++) {
      Eigen::AffineCompact2f transform;
      transform.setIdentity();
      transform.translation() = kd.corners[i].cast<Scalar>();

      transforms->observations.at(0)[last_keypoint_id] = transform;
      new_poses0[last_keypoint_id] = transform;

      last_keypoint_id++;
    }

    if (calib.intrinsics.size() > 1) {
      trackPoints(images.at(0), images.at(1), new_poses0, new_poses1);

      for (const auto& kv : new_poses1) {
        transforms->observations.at(1).emplace(kv);
      }
    }
  }

  void filterPoints() {
    if (calib.intrinsics.size() < 2) return;

    std::set<KeypointId> lm_to_remove;

    std::vector<KeypointId> kpid;
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

    for (const auto& kv : transforms->observations.at(1)) {
      auto it = transforms->observations.at(0).find(kv.first);

      if (it != transforms->observations.at(0).end()) {
        proj0.emplace_back(it->second.translation());
        proj1.emplace_back(kv.second.translation());
        kpid.emplace_back(kv.first);
      }
    }

    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    calib.intrinsics[1].unproject(proj1, p3d1, p3d1_success);

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error =
            std::abs(p3d0[i].transpose() * E * p3d1[i]);

        if (epipolar_error > config.optical_flow_epipolar_error) {
          lm_to_remove.emplace(kpid[i]);
        }
      } else {
        lm_to_remove.emplace(kpid[i]);
      }
    }

    for (int id : lm_to_remove) {
      transforms->observations.at(1).erase(id);
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  int64_t t_ns;

  size_t frame_counter;

  KeypointId last_keypoint_id;

  VioConfig config;
  basalt::Calibration<Scalar> calib;

  cv::Size win_size;
  cv::TermCriteria term_criteria;

  OpticalFlowResult::Ptr transforms;
  std::vector<cv::Mat> old_images, images;

  Matrix4 E;

  std::shared_ptr<std::thread> processing_thread;
};

}  // namespace basalt
