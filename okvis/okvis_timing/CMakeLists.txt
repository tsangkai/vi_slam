cmake_minimum_required(VERSION 2.8.11)

project(okvis_timing)

add_library(okvis_timing
  src/Timer.cpp
  src/NsecTimeUtilities.cpp
)

target_include_directories(okvis_timing
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE okvis_util
)

target_link_libraries(okvis_timing 
  PUBLIC okvis_util
)

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
