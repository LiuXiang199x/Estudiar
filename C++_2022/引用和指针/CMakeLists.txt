project(引用与指针)

cmake_minimum_required(VERSION 3.10)

include_directories(include)

aux_source_directory(src SRC_LIST)
add_executable(test ${SRC_LIST})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)