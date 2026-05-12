#include <ros2bag/ros2bag.h>

#include <sqlite3.h>

#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ros2bag {

namespace {

bool isDirectory(const std::string& path) {
  struct stat st {};
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool endsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

std::string findDb3File(const std::string& path) {
  if (!isDirectory(path)) return path;

  DIR* dir = opendir(path.c_str());
  if (!dir) throw std::runtime_error("Failed to open bag directory: " + path);

  std::vector<std::string> candidates;
  while (dirent* entry = readdir(dir)) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    if (endsWith(name, ".db3")) candidates.push_back(path + "/" + name);
  }
  closedir(dir);

  std::sort(candidates.begin(), candidates.end());
  if (candidates.empty()) {
    throw std::runtime_error("No rosbag2 sqlite3 .db3 file found in: " + path);
  }

  return candidates.front();
}

class SqliteStatement {
 public:
  SqliteStatement(sqlite3* db, const char* sql) {
    if (sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
      throw std::runtime_error(sqlite3_errmsg(db));
    }
  }

  ~SqliteStatement() {
    if (stmt_) sqlite3_finalize(stmt_);
  }

  sqlite3_stmt* get() { return stmt_; }

 private:
  sqlite3_stmt* stmt_ = nullptr;
};

class CdrReader {
 public:
  explicit CdrReader(const std::vector<uint8_t>& data) : data_(data) {
    if (data_.size() < 4) throw std::runtime_error("CDR buffer is too small");
    offset_ = kPayloadOffset;
  }

  int32_t readInt32() { return readPrimitive<int32_t>(); }
  uint32_t readUInt32() { return readPrimitive<uint32_t>(); }
  uint8_t readUInt8() { return readPrimitive<uint8_t>(); }
  double readDouble() { return readPrimitive<double>(); }

  std::string readString() {
    uint32_t len = readUInt32();
    if (len == 0) return "";
    require(len);

    std::string res(reinterpret_cast<const char*>(&data_[offset_]), len);
    offset_ += len;
    if (!res.empty() && res.back() == '\0') res.pop_back();
    return res;
  }

  std::vector<uint8_t> readUInt8Sequence() {
    uint32_t len = readUInt32();
    require(len);
    std::vector<uint8_t> res(data_.begin() + offset_,
                             data_.begin() + offset_ + len);
    offset_ += len;
    return res;
  }

 private:
  template <class T>
  T readPrimitive() {
    align(sizeof(T));
    require(sizeof(T));

    T value;
    std::memcpy(&value, data_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  void align(size_t alignment) {
    const size_t payload_offset = offset_ - kPayloadOffset;
    size_t padding = (alignment - (payload_offset % alignment)) % alignment;
    require(padding);
    offset_ += padding;
  }

  void require(size_t size) const {
    if (offset_ + size > data_.size()) {
      throw std::runtime_error("CDR buffer ended unexpectedly");
    }
  }

  const std::vector<uint8_t>& data_;
  static constexpr size_t kPayloadOffset = 4;
  size_t offset_ = 0;
};

Header readHeader(CdrReader& reader) {
  Header header;
  header.stamp.sec = reader.readInt32();
  header.stamp.nanosec = reader.readUInt32();
  header.frame_id = reader.readString();
  return header;
}

Vector3 readVector3(CdrReader& reader) {
  Vector3 v;
  v.x = reader.readDouble();
  v.y = reader.readDouble();
  v.z = reader.readDouble();
  return v;
}

Quaternion readQuaternion(CdrReader& reader) {
  Quaternion q;
  q.x = reader.readDouble();
  q.y = reader.readDouble();
  q.z = reader.readDouble();
  q.w = reader.readDouble();
  return q;
}

void readDoubleArray(CdrReader& reader, double* values, size_t size) {
  for (size_t i = 0; i < size; i++) values[i] = reader.readDouble();
}

}  // namespace

struct Reader::Impl {
  sqlite3* db = nullptr;
  std::vector<TopicInfo> topics;
  std::unordered_map<int64_t, TopicInfo> topics_by_id;
  std::unique_ptr<SqliteStatement> read_statement;
  bool row_ready = false;

  ~Impl() {
    read_statement.reset();
    if (db) sqlite3_close(db);
  }
};

Reader::Reader() : impl_(new Impl) {}

Reader::~Reader() = default;

void Reader::open(const std::string& path) {
  const std::string db_path = findDb3File(path);

  if (sqlite3_open_v2(db_path.c_str(), &impl_->db, SQLITE_OPEN_READONLY,
                      nullptr) != SQLITE_OK) {
    throw std::runtime_error("Failed to open rosbag2 sqlite database: " +
                             db_path);
  }

  SqliteStatement topic_stmt(
      impl_->db,
      "SELECT id, name, type, serialization_format FROM topics ORDER BY id;");

  while (sqlite3_step(topic_stmt.get()) == SQLITE_ROW) {
    TopicInfo topic;
    topic.id = sqlite3_column_int64(topic_stmt.get(), 0);
    topic.name =
        reinterpret_cast<const char*>(sqlite3_column_text(topic_stmt.get(), 1));
    topic.type =
        reinterpret_cast<const char*>(sqlite3_column_text(topic_stmt.get(), 2));
    topic.serialization_format =
        reinterpret_cast<const char*>(sqlite3_column_text(topic_stmt.get(), 3));

    impl_->topics.push_back(topic);
    impl_->topics_by_id[topic.id] = topic;
  }

  impl_->read_statement.reset(
      new SqliteStatement(impl_->db,
                          "SELECT topic_id, timestamp, data FROM messages "
                          "ORDER BY timestamp, id;"));
}

const std::vector<TopicInfo>& Reader::get_all_topics_and_types() const {
  return impl_->topics;
}

bool Reader::has_next() {
  if (impl_->row_ready) return true;

  int rc = sqlite3_step(impl_->read_statement->get());
  if (rc == SQLITE_ROW) {
    impl_->row_ready = true;
    return true;
  }
  if (rc == SQLITE_DONE) return false;

  throw std::runtime_error(sqlite3_errmsg(impl_->db));
}

std::shared_ptr<SerializedMessage> Reader::read_next() {
  if (!has_next()) return nullptr;

  sqlite3_stmt* stmt = impl_->read_statement->get();
  int64_t topic_id = sqlite3_column_int64(stmt, 0);
  auto topic_it = impl_->topics_by_id.find(topic_id);
  if (topic_it == impl_->topics_by_id.end()) {
    throw std::runtime_error("Message references unknown topic id");
  }

  auto msg = std::make_shared<SerializedMessage>();
  msg->topic_name = topic_it->second.name;
  msg->time_stamp = sqlite3_column_int64(stmt, 1);

  const void* blob = sqlite3_column_blob(stmt, 2);
  int size = sqlite3_column_bytes(stmt, 2);
  const auto* begin = static_cast<const uint8_t*>(blob);
  msg->data.assign(begin, begin + size);

  impl_->row_ready = false;
  return msg;
}

int64_t stampToNanoseconds(const Time& stamp) {
  return int64_t(stamp.sec) * 1000000000ll + int64_t(stamp.nanosec);
}

bool isType(const std::string& type, const std::string& ros2_type,
            const std::string& ros1_type) {
  return type == ros2_type || type == ros1_type;
}

template <>
std::shared_ptr<Image> deserialize<Image>(const SerializedMessage& bag_msg) {
  CdrReader reader(bag_msg.data);
  auto msg = std::make_shared<Image>();
  msg->header = readHeader(reader);
  msg->height = reader.readUInt32();
  msg->width = reader.readUInt32();
  msg->encoding = reader.readString();
  msg->is_bigendian = reader.readUInt8();
  msg->step = reader.readUInt32();
  msg->data = reader.readUInt8Sequence();
  return msg;
}

template <>
std::shared_ptr<Imu> deserialize<Imu>(const SerializedMessage& bag_msg) {
  CdrReader reader(bag_msg.data);
  auto msg = std::make_shared<Imu>();
  msg->header = readHeader(reader);
  msg->orientation = readQuaternion(reader);
  readDoubleArray(reader, msg->orientation_covariance, 9);
  msg->angular_velocity = readVector3(reader);
  readDoubleArray(reader, msg->angular_velocity_covariance, 9);
  msg->linear_acceleration = readVector3(reader);
  readDoubleArray(reader, msg->linear_acceleration_covariance, 9);
  return msg;
}

template <>
std::shared_ptr<TransformStamped> deserialize<TransformStamped>(
    const SerializedMessage& bag_msg) {
  CdrReader reader(bag_msg.data);
  auto msg = std::make_shared<TransformStamped>();
  msg->header = readHeader(reader);
  msg->child_frame_id = reader.readString();
  msg->transform.translation = readVector3(reader);
  msg->transform.rotation = readQuaternion(reader);
  return msg;
}

template <>
std::shared_ptr<PoseStamped> deserialize<PoseStamped>(
    const SerializedMessage& bag_msg) {
  CdrReader reader(bag_msg.data);
  auto msg = std::make_shared<PoseStamped>();
  msg->header = readHeader(reader);
  msg->pose.position = readVector3(reader);
  msg->pose.orientation = readQuaternion(reader);
  return msg;
}

template <>
std::shared_ptr<PointStamped> deserialize<PointStamped>(
    const SerializedMessage& bag_msg) {
  CdrReader reader(bag_msg.data);
  auto msg = std::make_shared<PointStamped>();
  msg->header = readHeader(reader);
  msg->point = readVector3(reader);
  return msg;
}

}  // namespace ros2bag
