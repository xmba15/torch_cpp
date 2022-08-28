cmake_minimum_required(VERSION 3.10)

function(__fetch_Pangolin download_module_path download_root)
  set(PANGOLIN_DOWNLOAD_ROOT ${download_root})
  configure_file(
    ${download_module_path}/Pangolin-download.cmake
    ${download_root}/CMakeLists.txt
    @ONLY
  )
  unset(PANGOLIN_DOWNLOAD_ROOT)

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY
      ${download_root}
  )
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY
      ${download_root}
  )

  add_subdirectory(
    ${download_root}/Pangolin-src
    ${download_root}/Pangolin-build
  )
endfunction()
