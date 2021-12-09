cmake_minimum_required (VERSION 3.5)

project (demo)

# find_library: 在指定目录下查找指定库，并把库的绝对路径存放到变量里，其第一个参数是变量名称，第二个参数是库名称，第三个参数是HINTS，第4个参数是路径，其它用法可以参考cmake文档
# target_link_libraries: 把目标文件与库文件进行链接
# 使用find_library的好处是在执行cmake ..时就会去查找库是否存在，这样可以提前发现错误，不用等到链接时。
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set (SRC_LIST ${PROJECT_SOURCE_DIR}/src/main.cc)

# find testFunc.h
include_directories (${PROJECT_SOURCE_DIR}/testFunc/inc)

find_library(TESTFUNC_LIB testFunc HINTS ${PROJECT_SOURCE_DIR}/testFunc/lib)

add_executable (main ${SRC_LIST})

target_link_libraries (main ${TESTFUNC_LIB})

# ps：在lib目录下有testFunc的静态库和动态库，find_library(TESTFUNC_LIB testFunc ...默认是查找动态库，如果想直接指定使用动态库还是静态库，可以写成find_library(TESTFUNC_LIB libtestFunc.so ...或者find_library(TESTFUNC_LIB libtestFunc.a ...
