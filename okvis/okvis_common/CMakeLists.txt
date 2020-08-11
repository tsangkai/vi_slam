cmake_minimum_required(VERSION 2.8.11)

project(okvis_common)

# build the library
add_library(okvis_common STATIC 
  src/VioInterface.cpp
  src/VioParametersReader.cpp
  include/okvis/FrameTypedefs.hpp
  include/okvis/Measurements.hpp
  include/okvis/Parameters.hpp
  include/okvis/Variables.hpp
  include/okvis/VioBackendInterface.hpp
  include/okvis/VioFrontendInterface.hpp
  include/okvis/VioInterface.hpp
  include/okvis/VioParametersReader.hpp
)

target_include_directories(okvis_common
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE Eigen3
)

target_include_directories(okvis_common
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# and link it                     
target_link_libraries(okvis_common
  PUBLIC okvis_util
  PUBLIC okvis_kinematics 
  PUBLIC okvis_time 
  PUBLIC okvis_cv
)
