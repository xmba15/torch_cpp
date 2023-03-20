cmake_minimum_required(VERSION 3.10)

project(torch_cpp VERSION 0.0.1)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(LIBRARY_NAME ${PROJECT_NAME})

set(LIBTORCH_DIR "/opt/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

find_package(CUDA QUIET)

if(CUDA_FOUND AND USE_GPU)
  add_definitions(-DENABLE_GPU=1)
else()
  add_definitions(-DENABLE_GPU=0)
endif()

add_compile_options(
  "$<$<CONFIG:Debug>:-DENABLE_DEBUG=1>"
  "$<$<CONFIG:Release>:-DENABLE_DEBUG=0>"
)

add_subdirectory(src)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)

if(BUILD_TEST)
  enable_testing()
  add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# installation
# ------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS ${LIBRARY_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/torch_cpp
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/torch_cpp
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/torch_cpp
)

set(EXPORT_CONFIG_FILE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake)
set(EXPORT_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME})
CONFIGURE_PACKAGE_CONFIG_FILE(
    ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${EXPORT_CONFIG_FILE}
    INSTALL_DESTINATION ${EXPORT_CONFIG_INSTALL_DIR}
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
install(EXPORT ${PROJECT_NAME}Targets
    DESTINATION ${EXPORT_CONFIG_INSTALL_DIR}
)
install(FILES ${EXPORT_CONFIG_FILE}
    DESTINATION ${EXPORT_CONFIG_INSTALL_DIR}
)
