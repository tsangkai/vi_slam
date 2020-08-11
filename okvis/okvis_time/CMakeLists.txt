cmake_minimum_required(VERSION 2.8.11)

project(okvis_time)

# build the library
add_library(okvis_time
  src/Time.cpp
  src/Duration.cpp)

target_include_directories(okvis_time
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(okvis_time
  PUBLIC okvis_util
)