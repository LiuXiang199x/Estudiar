project(test_cpp)

cmake_minimum_required(VERSION 3.10)

FIND_PACKAGE(OpenCV REQUIRED)


# link headers 链接头文件
INCLUDE_DIRECTORIES({OpenCV_INCLUDE_DIRS})

include_directories (../include)

# set (SRC_LIST main.cpp)
# 添加源代码文件到SRC_LIST变量中
AUX_SOURCE_DIRECTORY(src/. SRC_LIST)

add_executable (test_eigen ${SRC_LIST})

TARGET_LINK_LIBRARIES(test_eigen ${OpenCV_LIBS})

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
