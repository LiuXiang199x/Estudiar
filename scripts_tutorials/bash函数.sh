#!/usr/bin/env bash

hello() {
	echo "hello world"
	echo "hello world $1"   # $1 意味着调用第一个参数，这个参数是你在teminal中运行脚本时给的参数
}


# 参数变量, 注意这个不是函数
# $1~$9：函数的第一个到第9个的参数。
# $0：函数所在的脚本名。
# $#：函数的参数总数。
# $@：函数的全部参数，参数之间使用空格分隔。
# $*：函数的全部参数，参数之间使用变量$IFS值的第一个字符分隔，默认为空格，但是可以自定义。

function alias {   # 这里要空格！
	echo "alice: $@"
	echo "$0: $1 $2 $3 $4"
	echo "$# arguments"
}

# return不细说了
# Bash 函数体内直接声明的变量，属于全局变量，整个脚本都可以读取
foo=2   # 有些地方空格是一定要写的，像赋值的话，空格不要！！！
fn() {
	foo=22
}

fn

echo "foo = $foo"
