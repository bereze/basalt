#ifndef BASALT_THIRDPARTY_ROS2BAG_H
#define BASALT_THIRDPARTY_ROS2BAG_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ros2bag {

struct Time {
  int32_t sec = 0;
  uint32_t nanosec = 0;
};

struct Header {
  Time stamp;
  std::string frame_id;
};

struct Vector3 {
  double x = 0;
  double y = 0;
  double z = 0;
};

using Point = Vector3;

struct Quaternion {
  double x = 0;
  double y = 0;
  double z = 0;
  double w = 1;
};

struct Image {
  using SharedPtr = std::shared_ptr<Image>;
  using ConstSharedPtr = std::shared_ptr<const Image>;

  Header header;
  uint32_t height = 0;
  uint32_t width = 0;
  std::string encoding;
  uint8_t is_bigendian = 0;
  uint32_t step = 0;
  std::vector<uint8_t> data;
};

struct Imu {
  using SharedPtr = std::shared_ptr<Imu>;

  Header header;
  Quaternion orientation;
  double orientation_covariance[9] = {};
  Vector3 angular_velocity;
  double angular_velocity_covariance[9] = {};
  Vector3 linear_acceleration;
  double linear_acceleration_covariance[9] = {};
};

struct Transform {
  Vector3 translation;
  Quaternion rotation;
};

struct TransformStamped {
  using SharedPtr = std::shared_ptr<TransformStamped>;

  Header header;
  std::string child_frame_id;
  Transform transform;
};

struct Pose {
  Point position;
  Quaternion orientation;
};

struct PoseStamped {
  using SharedPtr = std::shared_ptr<PoseStamped>;

  Header header;
  Pose pose;
};

struct PointStamped {
  using SharedPtr = std::shared_ptr<PointStamped>;

  Header header;
  Point point;
};

struct TopicInfo {
  int64_t id = 0;
  std::string name;
  std::string type;
  std::string serialization_format;
};

struct SerializedMessage {
  std::string topic_name;
  int64_t time_stamp = 0;
  std::vector<uint8_t> data;
};

int64_t stampToNanoseconds(const Time& stamp);
bool isType(const std::string& type, const std::string& ros2_type,
            const std::string& ros1_type);

class Reader {
 public:
  Reader();
  ~Reader();

  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  void open(const std::string& path);
  const std::vector<TopicInfo>& get_all_topics_and_types() const;
  bool has_next();
  std::shared_ptr<SerializedMessage> read_next();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

template <class MsgT>
std::shared_ptr<MsgT> deserialize(const SerializedMessage& bag_msg);

template <>
std::shared_ptr<Image> deserialize<Image>(const SerializedMessage& bag_msg);

template <>
std::shared_ptr<Imu> deserialize<Imu>(const SerializedMessage& bag_msg);

template <>
std::shared_ptr<TransformStamped> deserialize<TransformStamped>(
    const SerializedMessage& bag_msg);

template <>
std::shared_ptr<PoseStamped> deserialize<PoseStamped>(
    const SerializedMessage& bag_msg);

template <>
std::shared_ptr<PointStamped> deserialize<PointStamped>(
    const SerializedMessage& bag_msg);

}  // namespace ros2bag

#endif  // BASALT_THIRDPARTY_ROS2BAG_H
