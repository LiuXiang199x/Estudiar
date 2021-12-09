aux_source_directory (. SRC_LIST)

# 这里又出现一个新的命令set，是用于定义变量的，EXECUTABLE_OUT_PATH和PROJECT_SOURCE_DIR是CMake自带的预定义变量，其意义如下，
# EXECUTABLE_OUTPUT_PATH ：目标二进制可执行文件的存放位置
# PROJECT_SOURCE_DIR：工程的根目录
# 所以，这里set的意思是把存放elf文件的位置设置为工程根目录下的bin目录。（cmake有很多预定义变量，详细的可以网上搜索一下）

include_directories (../include)

add_executable (main ${SRC_LIST})

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


