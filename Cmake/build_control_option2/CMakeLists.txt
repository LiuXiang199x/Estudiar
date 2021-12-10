cmake_minimum_required(VERSION 3.5)

# 可以通过定义宏来控制打印的信息，我们CMakeLists.txt内容如下
project(demo)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

option(WWW1 "print one message" OFF)
option(WWW2 "print another message" OFF)

if (WWW1)
    add_definitions(-DWWW1)
endif()

if (WWW2)
    add_definitions(-DWWW2)
endif()

add_executable(main main.cc)
# 这里把option的名字保持和main.c里的宏名称一致，这样更加直观，也可以选择不同的名字。通过与add_definitions()的配合，就可以控制单个bin文件的打印输出了
