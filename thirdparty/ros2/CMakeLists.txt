cmake_minimum_required(VERSION 3.10...3.18)

find_package(SQLite3 REQUIRED)
find_package(LZ4 REQUIRED)

add_library(ros2bag STATIC src/ros2bag.cpp)
target_compile_features(ros2bag PUBLIC cxx_std_17)

target_include_directories(ros2bag PUBLIC include)
# vendored foxglove/mcap reader only needs lz4 chunk decompression; zstd is not used
target_compile_definitions(ros2bag PRIVATE MCAP_COMPRESSION_NO_ZSTD)
target_link_libraries(ros2bag PUBLIC SQLite::SQLite3 LZ4::lz4)
