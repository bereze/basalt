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

#include <iostream>
#include <memory>

#include <Eigen/Geometry>

#include <basalt/utils/vio_config.h>

#include <basalt/io/dataset_io.h>
#include <basalt/calibration/calibration.hpp>
#include <basalt/camera/stereographic_param.hpp>
#include <basalt/utils/sophus_utils.hpp>
#include <basalt/utils/time_utils.hpp>

#include <tbb/concurrent_queue.h>

namespace basalt {

using KeypointId = size_t;

struct OpticalFlowInput {
  using Ptr = std::shared_ptr<OpticalFlowInput>;

  int64_t t_ns;
  std::vector<ImageData> img_data;
};

struct OpticalFlowResult {
  using Ptr = std::shared_ptr<OpticalFlowResult>;

  int64_t t_ns;
  std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>>
      observations;

  std::vector<std::map<KeypointId, size_t>> pyramid_levels;

  OpticalFlowInput::Ptr input_images;
};

class OpticalFlowBase {
 public:
  using Ptr = std::shared_ptr<OpticalFlowBase>;

  tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr> input_queue;
  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr>* output_queue = nullptr;

  Eigen::MatrixXf patch_coord;

 protected:
  // Number of processed frames between average-timing printouts.
  static constexpr size_t TIMING_PRINT_EVERY_N_FRAMES = 100;

  // Accumulate the time it took to process one frame and periodically print
  // the running average optical flow processing time.
  void recordFrameTime(double time_s) {
    total_frame_time_s_ += time_s;
    num_timed_frames_++;

    if (num_timed_frames_ % TIMING_PRINT_EVERY_N_FRAMES == 0) {
      std::cout << "[optical_flow] average processing time over "
                << num_timed_frames_ << " frames: "
                << 1000.0 * total_frame_time_s_ / num_timed_frames_
                << " ms/frame" << std::endl;
    }
  }

 private:
  double total_frame_time_s_ = 0;
  size_t num_timed_frames_ = 0;
};

class OpticalFlowFactory {
 public:
  static OpticalFlowBase::Ptr getOpticalFlow(const VioConfig& config,
                                             const Calibration<double>& cam);
};
}  // namespace basalt
