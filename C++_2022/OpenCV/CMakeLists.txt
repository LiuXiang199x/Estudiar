project(C_opencv)

cmake_minimum_required(VERSION 3.10)

find_library(OpenCV required)
include_directories({OpenCV_INCLUDE_DIRS})

add_executable (test src/main.cc)
TARGET_LINK_LIBRARIES(test ${OpenCV_LIBS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)