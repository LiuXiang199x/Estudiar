#!/usr/bin/env bash

# 函数返回值问题：
# 参数返回，可以显示加：return 返回，如果不加，将以最后一条命令运行结果，作为返回值
# echo来返回 值！！！
# return 结束一个函数。类似于break。return默认返回函数中最后一个命令状态值，也可以给定参数值(0-256)
# 如果没有return命令，函数将返回最后一个指令的退出状态值

F1(){
	A=1
	echo $A
}

F3(){
	B=3
	return 1
}

F2(){
	echo "2"
}



F1
F2
F3
