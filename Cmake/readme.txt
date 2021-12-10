https://blog.csdn.net/whahu1989/article/details/82078563

一 安装CMake
本文使用ubuntu18.04，安装cmake使用如下命令，
sudo apt install cmake
安装完成后，在终端下输入cmake -version查看cmake版本，

二 简单样例
src_solo

三 同一目录下多个源文件
接下来进入稍微复杂的例子：在同一个目录下有多个源文件。
src_2
src_multi


四 不同目录下多个源文件
一般来说，当程序文件比较多时，我们会进行分类管理，把代码根据功能放在不同的目录下，这样方便查找。那么这种情况下如何编写CMakeLists.txt呢？
我们把之前的源文件整理一下（新建2个目录test_func和test_func1），整理好后整体文件结构如下
differente_cato


五 正规一点的组织结构
正规一点来说，一般会把源文件放到src目录下，把头文件放入到include文件下，生成的对象文件放入到build目录下，最终输出的elf文件会放到bin目录下，这样整个结构更加清晰。让我们把前面的文件再次重新组织下: src_normal_estruc

六 动态库和静态库的编译控制
有时只需要编译出动态库和静态库，然后等着让其它程序去使用。让我们看下这种情况该如何使用cmake。首先按照如下重新组织文件，只留下testFunc.h和TestFunc.c:
Dynamic_Static_lib


七 对库进行链接
既然我们已经生成了库，那么就进行链接测试下。重新建一个工程目录，然后把上节生成的库拷贝过来，然后在在工程目录下新建src目录和bin目录，在src目录下添加一个main.c，整体结构如下
test.libs

八 添加编译选项
有时编译程序时想添加一些编译选项，如-Wall，-std=c++11等，就可以使用add_compile_options来进行操作。
build_option

九 添加控制选项
有时希望在编译代码时只编译一些指定的源码，可以使用cmake的option命令，主要遇到的情况分为2种：
	1. 本来要生成多个bin或库文件，现在只想生成部分指定的bin或库文件
	2. 对于同一个bin文件，只想编译其中部分代码（使用宏来控制）
build_control_option

十 总结
以上是自己学习CMake的一点学习记录，通过简单的例子让大家入门CMake，学习的同时也阅读了很多网友的博客。CMake的知识点还有很多，具体详情可以在网上搜索。总之，CMake可以让我们不用去编写复杂的Makefile，并且跨平台，是个非常强大并值得一学的工具。



