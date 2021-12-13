cmake_minimum_required (VERSION 2.8)

project (demo)

# 如果在同一目录下有多个源文件，那么只要在add_executable里把所有源文件都添加进去就可以了。但是如果有一百个源文件，再这样做就有点坑了，无法体现cmake的优越性，cmake提供了一个命令可以把指定目录下所有的源文件存储在一个变量中，这个命令就是 aux_source_directory(dir var)。
# 使用aux_source_directory把当前目录下的源文件存列表存放到变量SRC_LIST里，然后在add_executable里调用SRC_LIST（注意调用变量时的写法）。
aux_source_directory(. SRC_LIST)
add_executable(main ${SRC_LIST})




