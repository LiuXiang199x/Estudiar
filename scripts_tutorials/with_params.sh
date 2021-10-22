#!/usr/bin/env bash
#coding:utf-8

# $@返回一个全部参数的列表
echo "全部参数：" $@
echo "命令行参数数量：" $#
echo "$0 = " $0
echo "$1 = " $1
echo "$2 = " $2
echo "$3 = " $3

for i in $@;
do
	echo $i
done

