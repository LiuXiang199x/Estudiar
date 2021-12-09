cmake_minimum_required(VERSION 2.8)

project(demo)

# 这里出现了一个新的命令：include_directories。该命令是用来向工程添加多个指定头文件的搜索路径，路径之间用空格分隔。
include_directories(test_Func test_Func1)

# 如果在同一目录下有多个源文件，那么只要在add_executable里把所有源文件都添加进去就可以了。但是如果有一百个源文件，再这样做就有点坑了，无法体现cmake的优越性，cmake提供了一个命令可以把指定目录下所有的源文件存储在一个变量中，这个命令就是 aux_source_directory(dir var)。
# 使用aux_source_directory把当前目录下的源文件存列表存放到变量SRC_LIST里，然后在add_executable里调用SRC_LIST（注意调用变量时的写法）。
aux_source_directory (test_Func SRC_LIST)
aux_source_directory (test_Func1 SRC_LIST1)

add_executable (main main.cc ${SRC_LIST} ${SRC_LIST1})
