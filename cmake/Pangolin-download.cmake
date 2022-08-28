cmake_minimum_required(VERSION 3.10)

project(Pangolin-download NONE)

include(ExternalProject)

ExternalProject_Add(
  Pangolin
  SOURCE_DIR "@PANGOLIN_DOWNLOAD_ROOT@/Pangolin-src"
  BINARY_DIR "@PANGOLIN_DOWNLOAD_ROOT@/Pangolin-build"
  GIT_REPOSITORY
    https://github.com/stevenlovegrove/Pangolin
  GIT_TAG
    v0.8
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
