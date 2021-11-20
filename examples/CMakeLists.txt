cmake_minimum_required(VERSION 3.10)

add_executable(app
  ${CMAKE_CURRENT_LIST_DIR}/App.cpp
)

target_link_libraries(app
  PUBLIC
    ${LIBRARY_NAME}
)
