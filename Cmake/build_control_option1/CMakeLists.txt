cmake_minimum_required(VERSION 3.5)

project(demo)

option(MYDEBUG "enable debug compilation" OFF)

# 有时希望在编译代码时只编译一些指定的源码，可以使用cmake的option命令，主要遇到的情况分为2种：
### 1.本来要生成多个bin或库文件，现在只想生成部分指定的bin或库文件
### 2.对于同一个bin文件，只想编译其中部分代码（使用宏来控制）
# 假设我们现在的工程会生成2个bin文件，main1和main2

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(src)

# 直接cmake .. 只会编译出main1
# cmake .. -DMYDEBUG=ON  这样就可以编译出main1 和 main2了
