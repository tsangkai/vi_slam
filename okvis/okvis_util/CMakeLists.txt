cmake_minimum_required(VERSION 2.8.11)

project(okvis_util)

# nothing to build for now...
add_library(okvis_util INTERFACE)

target_include_directories(okvis_util
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

  

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/okvis" COMPONENT dev
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
