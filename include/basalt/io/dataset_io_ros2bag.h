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
#ifndef DATASET_IO_ROS2BAG_H
#define DATASET_IO_ROS2BAG_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <basalt/io/dataset_io.h>
#include <basalt/utils/filesystem.h>

#include <ros2bag/ros2bag.h>

namespace basalt {

class Ros2bagVioDataset : public VioDataset {
  using ImageMsgPtr = ros2bag::Image::ConstSharedPtr;

  size_t num_cams = 0;

  std::vector<int64_t> image_timestamps;

  // vector of images for every timestamp
  // assumes vectors size is num_cams for every timestamp with null pointers for
  // missing frames
  std::unordered_map<int64_t, std::vector<ImageMsgPtr>> image_data;

  Eigen::aligned_vector<AccelData> accel_data;
  Eigen::aligned_vector<GyroData> gyro_data;

  std::vector<int64_t> gt_timestamps;  // ordered gt timestamps
  Eigen::aligned_vector<Sophus::SE3d>
      gt_pose_data;  // TODO: change to eigen aligned

  int64_t mocap_to_imu_offset_ns = 0;

 public:
  ~Ros2bagVioDataset() {}

  size_t get_num_cams() const { return num_cams; }

  std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }

  const Eigen::aligned_vector<AccelData> &get_accel_data() const {
    return accel_data;
  }
  const Eigen::aligned_vector<GyroData> &get_gyro_data() const {
    return gyro_data;
  }
  const std::vector<int64_t> &get_gt_timestamps() const {
    return gt_timestamps;
  }
  const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const {
    return gt_pose_data;
  }

  int64_t get_mocap_to_imu_offset_ns() const { return mocap_to_imu_offset_ns; }

  std::vector<ImageData> get_image_data(int64_t t_ns) {
    std::vector<ImageData> res(num_cams);

    auto it = image_data.find(t_ns);

    if (it != image_data.end())
      for (size_t i = 0; i < num_cams; i++) {
        ImageData &id = res[i];
        const ImageMsgPtr &img_msg = it->second[i];

        if (!img_msg) continue;

        id.img.reset(
            new ManagedImage<uint16_t>(img_msg->width, img_msg->height));

        if (!img_msg->header.frame_id.empty() &&
            std::isdigit(
                static_cast<unsigned char>(img_msg->header.frame_id[0]))) {
          id.exposure = std::stol(img_msg->header.frame_id) * 1e-9;
        } else {
          id.exposure = -1;
        }

        if (img_msg->encoding == "mono8") {
          const uint8_t *data_in = img_msg->data.data();
          uint16_t *data_out = id.img->ptr;

          for (size_t i = 0; i < img_msg->data.size(); i++) {
            int val = data_in[i];
            val = val << 8;
            data_out[i] = val;
          }

        } else if (img_msg->encoding == "mono16") {
          std::memcpy(id.img->ptr, img_msg->data.data(), img_msg->data.size());
        } else {
          std::cerr << "Encoding " << img_msg->encoding << " is not supported."
                    << std::endl;
          std::abort();
        }
      }

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend class Ros2bagIO;
};

class Ros2bagIO : public DatasetIoInterface {
 public:
  Ros2bagIO() {}

  void read(const std::string &path) {
    if (!fs::exists(path))
      std::cerr << "No dataset found in " << path << std::endl;

    data.reset(new Ros2bagVioDataset);

    ros2bag::Reader reader;
    reader.open(path);

    std::map<std::string, std::string> topic_types;
    for (const auto &topic_metadata : reader.get_all_topics_and_types()) {
      topic_types[topic_metadata.name] = topic_metadata.type;
    }

    // Check serialization formats
    std::cout << "Serialization formats:" << std::endl;
    for (const auto &topic_metadata : reader.get_all_topics_and_types()) {
      std::cout << topic_metadata.name << ": " << topic_metadata.serialization_format << std::endl;
      if (topic_metadata.serialization_format != "cdr") {
        throw std::runtime_error("Unsupported serialization format: " + topic_metadata.serialization_format + " for topic " + topic_metadata.name + ". Only 'cdr' is supported.");
      }
    }

    std::set<std::string> cam_topics;
    std::string imu_topic;
    std::string mocap_topic;
    std::string point_topic;

    for (const auto &topic_type : topic_types) {
      const std::string &topic = topic_type.first;
      const std::string &type = topic_type.second;

      if (ros2bag::isType(type, "sensor_msgs/msg/Image", "sensor_msgs/Image")) {
        cam_topics.insert(topic);
      } else if (ros2bag::isType(type, "sensor_msgs/msg/Imu",
                                 "sensor_msgs/Imu") &&
                 topic.rfind("/fcu", 0) != 0) {
        imu_topic = topic;
      } else if (ros2bag::isType(type, "geometry_msgs/msg/TransformStamped",
                                 "geometry_msgs/TransformStamped") ||
                 ros2bag::isType(type, "geometry_msgs/msg/PoseStamped",
                                 "geometry_msgs/PoseStamped")) {
        mocap_topic = topic;
      } else if (ros2bag::isType(type, "geometry_msgs/msg/PointStamped",
                                 "geometry_msgs/PointStamped")) {
        point_topic = topic;
      }
    }

    std::cout << "imu_topic: " << imu_topic << std::endl;
    std::cout << "mocap_topic: " << mocap_topic << std::endl;
    std::cout << "cam_topics: ";
    for (const std::string &s : cam_topics) std::cout << s << " ";
    std::cout << std::endl;

    std::map<std::string, int> topic_to_id;
    int idx = 0;
    for (const std::string &s : cam_topics) {
      topic_to_id[s] = idx;
      idx++;
    }

    data->num_cams = cam_topics.size();

    int num_msgs = 0;

    int64_t min_time = std::numeric_limits<int64_t>::max();
    int64_t max_time = std::numeric_limits<int64_t>::min();

    std::vector<ros2bag::TransformStamped::SharedPtr> mocap_msgs;
    std::vector<ros2bag::PointStamped::SharedPtr> point_msgs;

    std::vector<int64_t>
        system_to_imu_offset_vec;  // t_imu = t_system + system_to_imu_offset
    std::vector<int64_t> system_to_mocap_offset_vec;  // t_mocap = t_system +
                                                      // system_to_mocap_offset

    std::set<int64_t> image_timestamps;

    while (reader.has_next()) {
      auto bag_msg = reader.read_next();
      const std::string &topic = bag_msg->topic_name;

      if (cam_topics.find(topic) != cam_topics.end()) {
        auto img_msg = ros2bag::deserialize<ros2bag::Image>(*bag_msg);
        int64_t timestamp_ns =
            ros2bag::stampToNanoseconds(img_msg->header.stamp);

        auto &img_vec = data->image_data[timestamp_ns];
        if (img_vec.size() == 0) img_vec.resize(data->num_cams);

        img_vec[topic_to_id.at(topic)] = img_msg;
        image_timestamps.insert(timestamp_ns);

        min_time = std::min(min_time, timestamp_ns);
        max_time = std::max(max_time, timestamp_ns);
      }

      if (imu_topic == topic) {
        auto imu_msg = ros2bag::deserialize<ros2bag::Imu>(*bag_msg);
        int64_t time = ros2bag::stampToNanoseconds(imu_msg->header.stamp);

        data->accel_data.emplace_back();
        data->accel_data.back().timestamp_ns = time;
        data->accel_data.back().data = Eigen::Vector3d(
            imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y,
            imu_msg->linear_acceleration.z);

        data->gyro_data.emplace_back();
        data->gyro_data.back().timestamp_ns = time;
        data->gyro_data.back().data = Eigen::Vector3d(
            imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
            imu_msg->angular_velocity.z);

        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);

        int64_t msg_arrival_time = bag_msg->time_stamp;
        system_to_imu_offset_vec.push_back(time - msg_arrival_time);
      }

      if (mocap_topic == topic) {
        ros2bag::TransformStamped::SharedPtr mocap_msg;

        if (ros2bag::isType(topic_types.at(topic),
                            "geometry_msgs/msg/TransformStamped",
                            "geometry_msgs/TransformStamped")) {
          mocap_msg = ros2bag::deserialize<ros2bag::TransformStamped>(*bag_msg);
        } else {
          auto pose_msg = ros2bag::deserialize<ros2bag::PoseStamped>(*bag_msg);

          mocap_msg = std::make_shared<ros2bag::TransformStamped>();
          mocap_msg->header = pose_msg->header;
          mocap_msg->transform.rotation = pose_msg->pose.orientation;
          mocap_msg->transform.translation.x = pose_msg->pose.position.x;
          mocap_msg->transform.translation.y = pose_msg->pose.position.y;
          mocap_msg->transform.translation.z = pose_msg->pose.position.z;
        }

        int64_t time = ros2bag::stampToNanoseconds(mocap_msg->header.stamp);

        mocap_msgs.push_back(mocap_msg);

        int64_t msg_arrival_time = bag_msg->time_stamp;
        system_to_mocap_offset_vec.push_back(time - msg_arrival_time);
      }

      if (point_topic == topic) {
        auto point_msg = ros2bag::deserialize<ros2bag::PointStamped>(*bag_msg);

        int64_t time = ros2bag::stampToNanoseconds(point_msg->header.stamp);

        point_msgs.push_back(point_msg);

        int64_t msg_arrival_time = bag_msg->time_stamp;
        system_to_mocap_offset_vec.push_back(time - msg_arrival_time);
      }

      num_msgs++;
    }

    data->image_timestamps.clear();
    data->image_timestamps.insert(data->image_timestamps.begin(),
                                  image_timestamps.begin(),
                                  image_timestamps.end());

    if (system_to_mocap_offset_vec.size() > 0 &&
        system_to_imu_offset_vec.size() > 0) {
      int64_t system_to_imu_offset =
          system_to_imu_offset_vec[system_to_imu_offset_vec.size() / 2];

      int64_t system_to_mocap_offset =
          system_to_mocap_offset_vec[system_to_mocap_offset_vec.size() / 2];

      data->mocap_to_imu_offset_ns =
          system_to_imu_offset - system_to_mocap_offset;
    }

    data->gt_pose_data.clear();
    data->gt_timestamps.clear();

    if (!mocap_msgs.empty())
      for (size_t i = 0; i < mocap_msgs.size() - 1; i++) {
        auto mocap_msg = mocap_msgs[i];

        int64_t time = ros2bag::stampToNanoseconds(mocap_msg->header.stamp);

        Eigen::Quaterniond q(
            mocap_msg->transform.rotation.w, mocap_msg->transform.rotation.x,
            mocap_msg->transform.rotation.y, mocap_msg->transform.rotation.z);

        Eigen::Vector3d t(mocap_msg->transform.translation.x,
                          mocap_msg->transform.translation.y,
                          mocap_msg->transform.translation.z);

        int64_t timestamp_ns = time + data->mocap_to_imu_offset_ns;
        data->gt_timestamps.emplace_back(timestamp_ns);
        data->gt_pose_data.emplace_back(q, t);
      }

    if (!point_msgs.empty())
      for (size_t i = 0; i < point_msgs.size() - 1; i++) {
        auto point_msg = point_msgs[i];

        int64_t time = ros2bag::stampToNanoseconds(point_msg->header.stamp);

        Eigen::Vector3d t(point_msg->point.x, point_msg->point.y,
                          point_msg->point.z);

        int64_t timestamp_ns = time;  // + data->mocap_to_imu_offset_ns;
        data->gt_timestamps.emplace_back(timestamp_ns);
        data->gt_pose_data.emplace_back(Sophus::SO3d(), t);
      }

    std::cout << "Total number of messages: " << num_msgs << std::endl;
    std::cout << "Image size: " << data->image_data.size() << std::endl;

    std::cout << "Min time: " << min_time << " max time: " << max_time
              << " mocap to imu offset: " << data->mocap_to_imu_offset_ns
              << std::endl;

    std::cout << "Number of mocap poses: " << data->gt_timestamps.size()
              << std::endl;
  }

  void reset() { data.reset(); }

  VioDatasetPtr get_data() {
    // return std::dynamic_pointer_cast<VioDataset>(data);
    return data;
  }

 private:
  std::shared_ptr<Ros2bagVioDataset> data;
};

}  // namespace basalt

#endif  // DATASET_IO_ROS2BAG_H
