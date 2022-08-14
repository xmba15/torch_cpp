cmake_minimum_required(VERSION 3.10)

add_executable(visualize_superpoint_app
  ${CMAKE_CURRENT_LIST_DIR}/App.cpp
)

target_link_libraries(visualize_superpoint_app
  PUBLIC
    ${LIBRARY_NAME}
)
