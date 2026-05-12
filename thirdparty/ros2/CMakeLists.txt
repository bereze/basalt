cmake_minimum_required(VERSION 3.10...3.18)

find_package(SQLite3 REQUIRED)

add_library(ros2bag STATIC src/ros2bag.cpp)
target_compile_features(ros2bag PUBLIC cxx_std_17)

target_include_directories(ros2bag PUBLIC include)
target_link_libraries(ros2bag PUBLIC SQLite::SQLite3)
