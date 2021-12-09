cmake_minimum_required (VERSION 2.8)

project (demo)

# 使用aux_source_directory把当前目录下的源文件存列表存放到变量SRC_LIST里，然后在add_executable里调用SRC_LIST（注意调用变量时的写法）。
aux_source_directory(. SRC_LIST)
add_executable(main ${SRC_LIST})
