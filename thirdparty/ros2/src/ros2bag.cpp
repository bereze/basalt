#include <ros2bag/ros2bag.h>

#include <sqlite3.h>

#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>

#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <optional>
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

enum class BagFormat { Sqlite3, Mcap };

struct ResolvedBagFile {
  BagFormat format;
  std::string path;
};

ResolvedBagFile resolveBagFile(const std::string& path) {
  if (!isDirectory(path)) {
    if (endsWith(path, ".mcap")) return {BagFormat::Mcap, path};
    if (endsWith(path, ".db3")) return {BagFormat::Sqlite3, path};
    throw std::runtime_error("Unrecognized rosbag2 file extension: " + path);
  }

  DIR* dir = opendir(path.c_str());
  if (!dir) throw std::runtime_error("Failed to open bag directory: " + path);

  std::vector<std::string> db3_candidates;
  std::vector<std::string> mcap_candidates;
  while (dirent* entry = readdir(dir)) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    if (endsWith(name, ".db3")) db3_candidates.push_back(path + "/" + name);
    if (endsWith(name, ".mcap")) mcap_candidates.push_back(path + "/" + name);
  }
  closedir(dir);

  std::sort(db3_candidates.begin(), db3_candidates.end());
  std::sort(mcap_candidates.begin(), mcap_candidates.end());

  // A rosbag2 directory is normally produced by a single storage plugin, so
  // finding candidates of both kinds should not happen in practice; prefer
  // .mcap deterministically if it ever does.
  if (!mcap_candidates.empty()) return {BagFormat::Mcap, mcap_candidates.front()};
  if (!db3_candidates.empty()) return {BagFormat::Sqlite3, db3_candidates.front()};

  throw std::runtime_error("No rosbag2 .db3 or .mcap file found in: " + path);
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

// Common interface implemented by each storage backend (sqlite3 or mcap), so
// that ros2bag::Reader can stay agnostic of the underlying container format.
struct BackendReader {
  virtual ~BackendReader() = default;
  virtual const std::vector<TopicInfo>& topics() const = 0;
  virtual bool has_next() = 0;
  virtual std::shared_ptr<SerializedMessage> read_next() = 0;
};

class Sqlite3BackendReader : public BackendReader {
 public:
  explicit Sqlite3BackendReader(const std::string& db_path) {
    if (sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READONLY,
                        nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to open rosbag2 sqlite database: " +
                               db_path);
    }

    SqliteStatement topic_stmt(
        db_,
        "SELECT id, name, type, serialization_format FROM topics ORDER BY id;");

    while (sqlite3_step(topic_stmt.get()) == SQLITE_ROW) {
      TopicInfo topic;
      topic.id = sqlite3_column_int64(topic_stmt.get(), 0);
      topic.name = reinterpret_cast<const char*>(
          sqlite3_column_text(topic_stmt.get(), 1));
      topic.type = reinterpret_cast<const char*>(
          sqlite3_column_text(topic_stmt.get(), 2));
      topic.serialization_format = reinterpret_cast<const char*>(
          sqlite3_column_text(topic_stmt.get(), 3));

      topics_.push_back(topic);
      topics_by_id_[topic.id] = topic;
    }

    read_statement_.reset(new SqliteStatement(
        db_, "SELECT topic_id, timestamp, data FROM messages "
             "ORDER BY timestamp, id;"));
  }

  ~Sqlite3BackendReader() override {
    read_statement_.reset();
    if (db_) sqlite3_close(db_);
  }

  const std::vector<TopicInfo>& topics() const override { return topics_; }

  bool has_next() override {
    if (row_ready_) return true;

    int rc = sqlite3_step(read_statement_->get());
    if (rc == SQLITE_ROW) {
      row_ready_ = true;
      return true;
    }
    if (rc == SQLITE_DONE) return false;

    throw std::runtime_error(sqlite3_errmsg(db_));
  }

  std::shared_ptr<SerializedMessage> read_next() override {
    if (!has_next()) return nullptr;

    sqlite3_stmt* stmt = read_statement_->get();
    int64_t topic_id = sqlite3_column_int64(stmt, 0);
    auto topic_it = topics_by_id_.find(topic_id);
    if (topic_it == topics_by_id_.end()) {
      throw std::runtime_error("Message references unknown topic id");
    }

    auto msg = std::make_shared<SerializedMessage>();
    msg->topic_name = topic_it->second.name;
    msg->time_stamp = sqlite3_column_int64(stmt, 1);

    const void* blob = sqlite3_column_blob(stmt, 2);
    int size = sqlite3_column_bytes(stmt, 2);
    const auto* begin = static_cast<const uint8_t*>(blob);
    msg->data.assign(begin, begin + size);

    row_ready_ = false;
    return msg;
  }

 private:
  sqlite3* db_ = nullptr;
  std::vector<TopicInfo> topics_;
  std::unordered_map<int64_t, TopicInfo> topics_by_id_;
  std::unique_ptr<SqliteStatement> read_statement_;
  bool row_ready_ = false;
};

class McapBackendReader : public BackendReader {
 public:
  explicit McapBackendReader(const std::string& path) {
    const mcap::Status open_status = reader_.open(path);
    if (!open_status.ok()) {
      throw std::runtime_error("Failed to open mcap file '" + path +
                               "': " + open_status.message);
    }

    const mcap::Status summary_status =
        reader_.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
    if (!summary_status.ok()) {
      std::cerr << "Warning: failed to read mcap summary for '" << path
                << "': " << summary_status.message << std::endl;
    }

    for (const auto& channel_entry : reader_.channels()) {
      const auto& channel = channel_entry.second;

      TopicInfo topic;
      topic.id = channel->id;
      topic.name = channel->topic;
      topic.serialization_format = channel->messageEncoding;
      if (mcap::SchemaPtr schema = reader_.schema(channel->schemaId)) {
        topic.type = schema->name;
      }
      topics_.push_back(topic);
    }

    mcap::ReadMessageOptions options;
    options.readOrder = hasUsableMessageIndex()
                             ? mcap::ReadMessageOptions::ReadOrder::LogTimeOrder
                             : mcap::ReadMessageOptions::ReadOrder::FileOrder;
    if (options.readOrder == mcap::ReadMessageOptions::ReadOrder::FileOrder) {
      std::cerr << "Warning: mcap file '" << path
                << "' has no usable message index; reading in file order, "
                   "which may not be timestamp-sorted."
                << std::endl;
    }

    view_.emplace(reader_.readMessages(
        [](const mcap::Status& status) {
          std::cerr << "mcap read warning: " << status.message << std::endl;
        },
        options));
    it_.emplace(view_->begin());
    end_.emplace(view_->end());
  }

  const std::vector<TopicInfo>& topics() const override { return topics_; }

  bool has_next() override { return *it_ != *end_; }

  std::shared_ptr<SerializedMessage> read_next() override {
    if (!has_next()) return nullptr;

    mcap::LinearMessageView::Iterator& it = *it_;
    const mcap::MessageView& mv = *it;

    auto msg = std::make_shared<SerializedMessage>();
    msg->topic_name = mv.channel->topic;
    msg->time_stamp = static_cast<int64_t>(mv.message.logTime);

    // mcap::Message::data is only valid until the iterator is advanced, so
    // the payload must be copied out before incrementing it below.
    const auto* begin = reinterpret_cast<const uint8_t*>(mv.message.data);
    msg->data.assign(begin, begin + mv.message.dataSize);

    ++it;
    return msg;
  }

 private:
  bool hasUsableMessageIndex() const {
    const auto& chunk_indexes = reader_.chunkIndexes();
    return std::any_of(
        chunk_indexes.begin(), chunk_indexes.end(),
        [](const mcap::ChunkIndex& ci) { return ci.messageIndexLength > 0; });
  }

  mcap::McapReader reader_;
  std::vector<TopicInfo> topics_;
  std::optional<mcap::LinearMessageView> view_;
  std::optional<mcap::LinearMessageView::Iterator> it_;
  std::optional<mcap::LinearMessageView::Iterator> end_;
};

}  // namespace

struct Reader::Impl {
  std::unique_ptr<BackendReader> backend;
};

Reader::Reader() : impl_(new Impl) {}

Reader::~Reader() = default;

void Reader::open(const std::string& path) {
  const ResolvedBagFile resolved = resolveBagFile(path);
  switch (resolved.format) {
    case BagFormat::Sqlite3:
      impl_->backend.reset(new Sqlite3BackendReader(resolved.path));
      break;
    case BagFormat::Mcap:
      impl_->backend.reset(new McapBackendReader(resolved.path));
      break;
  }
}

const std::vector<TopicInfo>& Reader::get_all_topics_and_types() const {
  return impl_->backend->topics();
}

bool Reader::has_next() { return impl_->backend->has_next(); }

std::shared_ptr<SerializedMessage> Reader::read_next() {
  return impl_->backend->read_next();
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
std::shared_ptr<CompressedImage> deserialize<CompressedImage>(
    const SerializedMessage& bag_msg) {
  CdrReader reader(bag_msg.data);
  auto msg = std::make_shared<CompressedImage>();
  msg->header = readHeader(reader);
  msg->format = reader.readString();
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
