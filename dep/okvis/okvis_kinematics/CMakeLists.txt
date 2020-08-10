cmake_minimum_required(VERSION 2.8.11)

project(okvis_kinematics)

# require Eigen3
find_package( Eigen REQUIRED )

# build the library
add_library(${PROJECT_NAME} STATIC src/dependency-tracker.cc)


target_include_directories(okvis_kinematics
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE okvis_util
  PRIVATE Eigen)

target_include_directories(okvis_kinematics
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")




# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")

# testing
if(BUILD_TESTS)
  if(APPLE)
    add_definitions(-DGTEST_HAS_TR1_TUPLE=1)
  else()
    add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
  endif(APPLE)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  add_executable(${PROJECT_TEST_NAME}
    test/runTests.cpp
    test/TestTransformation.cpp
  )
  target_link_libraries(${PROJECT_TEST_NAME} 
    ${PROJECT_NAME} 
    ${GTEST_LIBRARY} 
    pthread)
  add_test(test ${PROJECT_TEST_NAME})
endif()
