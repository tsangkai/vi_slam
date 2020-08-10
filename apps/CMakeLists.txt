add_executable(okvis_app_synchronous okvis_app_synchronous.cpp)

target_link_libraries(okvis_app_synchronous PRIVATE
  okvis_util
  okvis_kinematics
  okvis_time
  okvis_cv 
  okvis_common
  okvis_ceres
  okvis_timing
  okvis_matcher
  okvis_frontend 
  okvis_multisensor_processing
  pthread 
  Boost::filesystem
  Boost::system
  )