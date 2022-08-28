cmake_minimum_required(VERSION 3.10)

find_package(Pangolin QUIET)
if (NOT Pangolin_FOUND)
  message(STATUS "Pangolin is not found")
  return()
endif()

add_executable(visualize_pose_ground_truth
  ${CMAKE_CURRENT_LIST_DIR}/VisualizePoseGroundTruth.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)

target_link_libraries(visualize_pose_ground_truth
  PUBLIC
    ${OpenCV_LIBRARIES}
    ${Pangolin_LIBRARIES}
)

target_include_directories(visualize_pose_ground_truth
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

add_executable(visual_odometry_app
  ${CMAKE_CURRENT_LIST_DIR}/App.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)

target_link_libraries(visual_odometry_app
  PUBLIC
    ${LIBRARY_NAME}
)
