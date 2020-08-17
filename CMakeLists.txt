cmake_minimum_required(VERSION 3.8)

# set the project name
project(vi_slam)

# options for build configuration
option (USE_SYSTEM_BRISK
        "Use brisk via find_package rather than downloading it as part of okvis" OFF) 
option (USE_SYSTEM_CERES
        "Use ceres via find_package rather than downloading it as part of okvis" OFF) 


set(N_CORES 3 CACHE STRING "Using N number of cores for parallel build")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++11 -fPIC")

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  message(STATUS "Assuming SSE instructions available.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
endif()

get_directory_property(defs COMPILE_DEFINITIONS)





# EXTERNAL DEPENDENCIES
include(ExternalProject)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include_directories(${CMAKE_BINARY_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/lib)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})   # supposed to be global

# libvisensor
find_package(VISensor)

if(VISENSORDRIVER_FOUND)
  add_definitions(-DHAVE_LIBVISENSOR)
  set(LIBVISENSOR TRUE)
  message(STATUS "Found libvisensor. Setting HAVE_LIBVISENSOR flag.")
  include_directories(${VISensorDriver_INCLUDE_DIR})
  # install find script
  install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindVISensor.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/CMake)
else()
  message(STATUS "libvisensor not found")
endif()


# glog
find_package(Glog REQUIRED)
include_directories(BEFORE ${GLOG_INCLUDE_DIRS}) 


# brisk
find_package(brisk 2 REQUIRED)
include_directories(BEFORE ${BRISK_INCLUDE_DIRS})
message(STATUS "Using system brisk. Found at ${BRISK_INCLUDE_DIRS}.")


# ceres
find_package(SuiteSparse REQUIRED QUIET) #will be needed in any case...
find_package(CXSparse QUIET) #optional

find_package(Ceres 1.9 REQUIRED)
include_directories(BEFORE ${CERES_INCLUDE_DIRS})
message(STATUS "Using system Ceres.")






# OpenGV
set(CMAKE_CXX_FLAGS_WSUPPRESS 
    "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-maybe-uninitialized -Wno-sign-compare -Wno-unused-but-set-variable -Wno-unused-variable -Wno-pedantic")
message(STATUS "NOTE: Suppressing some warnings when compiling OpenGV.")
ExternalProject_Add(opengv_external
    GIT_REPOSITORY https://github.com/laurentkneip/opengv
    GIT_TAG cc32b16281aa6eab67cb28a61cf87a2a5c2b0961
    UPDATE_COMMAND ""
    PATCH_COMMAND # forcing static libs
        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/cmake/opengv/CMakeLists.txt ${CMAKE_CURRENT_BINARY_DIR}/opengv/src/opengv/CMakeLists.txt
        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/cmake/opengv/opengvConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/opengv/src/opengv/opengvConfig.cmake.in
        COMMENT "Forcing our own CMakeLists.txt to build OpenGV (static library support)."
    CMAKE_ARGS 
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> 
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE} 
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_WSUPPRESS} # suppress warnings...
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/opengv
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/opengv/src/opengv
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    BUILD_IN_SOURCE 0
    BUILD_COMMAND make -j${N_CORES}
    INSTALL_COMMAND make install
)
add_library(opengv STATIC IMPORTED)
set(OpenGV_LIBRARIES opengv)
set_target_properties(opengv PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/libopengv.a)
add_dependencies(${OpenGV_LIBRARIES} opengv_external)

find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS}) 


# sub-directory

add_subdirectory(okvis)
add_subdirectory(apps)
