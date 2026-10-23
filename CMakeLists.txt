cmake_minimum_required(VERSION 3.13)
project(mhd3d LANGUAGES CXX)

# Path to the existing BOUT++ build tree. Override on the command line with:
#   cmake -S . -B build -DBOUT_BUILD_DIR=/some/other/build
set(BOUT_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../BOUT-dev/build0"
    CACHE PATH "BOUT++ build or install tree")

find_package(bout++ REQUIRED
  PATHS "${BOUT_BUILD_DIR}"
  NO_DEFAULT_PATH)          # don't accidentally pick up a system BOUT++

message(STATUS "Found BOUT++ ${bout++_VERSION} at ${BOUT_BUILD_DIR}")

add_executable(mhd3d mhd3d.cxx)
target_link_libraries(mhd3d PRIVATE bout++::bout++)

# Make ./data/BOUT.inp visible from the build directory
add_custom_command(TARGET mhd3d POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${CMAKE_CURRENT_SOURCE_DIR}/data" "$<TARGET_FILE_DIR:mhd3d>/data")
