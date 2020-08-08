cmake_minimum_required(VERSION 3.8)

# set the project name
project(vi_slam)


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")



# Boost
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS}) 

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) 

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR}) 

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS}) 


# 
add_subdirectory(dep)
# add_subdirectory(src)


# add the executable
add_executable(okvis_app_synchronous.o okvis_app_synchronous.cpp)



# target_link_libraries(okvis_app_synchronous.o
#  ${Boost_LIBRARIES}
#  ${OpenCV_LIBRARIES}
#  ${EIGEN3_LIBRARIES}
#  okvis_time
#  okvis_common
#  )


