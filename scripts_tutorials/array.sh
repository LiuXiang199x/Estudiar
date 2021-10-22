#!/usr/bin/env bash

# 单个赋值方法创建数组
ARRAY1[0]=1

# 读取数组不是这么读的！
echo "ARRAY1 = $ARRAY1"  # ARRAY1 = 1

# 一次性赋值创建数组
ARRAY2=(1 2 3)
ARRAY3=([2]=1 [0]=11 [1]=111)

# 数组的读取方式: 全部：@ 或 *
echo "ARRAY2[ALL] = ${ARRAY2[@]}"  # ARRAY2[ALL] = 1 2 3
for i in ${ARRAY2[@]}
do
	echo $i
done
echo "ARRAY2[0] = ${ARRAY2[0]}"   # ARRAY2[0] = 1
echo "ARRAY3[0] = ${ARRAY3[0]}"   # ARRAY3[0] = 11
echo "ARRAY3[-1] = ${ARRAY3[-1]}"  # ARRAY3[-1] = 1

# 拷贝一个数组
a=( ${ARRAY2[@]} )
echo "a: ${a[@]}"   # a: 1 2 3

# append 方法
b=(${ARRAY2[@]} 4)
c=(${ARRAY2[1]} 4)
echo "b: ${b[@]}"   # 1 2 3 4
echo "c: ${c[@]}"   # 2 4

# 数组的长度 ${#array[*]}  或  ${#array[@]}
d[100]=foo  # 把字符串赋值给100位置的数组元素，这时的数组只有一个元素。
echo "{#d[@]}: ${#d[@]}"  # 1
echo "{#d[100]}: ${#d[100]}"  # 3 返回数组第100号成员a[100]的值（foo）的字符串长度。

# 提取数组序号
e=( [5]=a [9]=b [23]=c )
echo "!a[@]: ${!a[@]}"
for i in ${!a[*]}; do
	echo "i: $i ${e[$i]}"  # 0 1 2 可以看出来想通过0 1 2来引用数组元素是不行的，因为他们位置给定律
	echo "i: $i ${e[5]}"    # 0 a; 1 a; 2 a;
done

# 提取数组成员 ${array[@]:position:length}, 注意这里并不是切片！
food=( apples bananas cucumbers dates eggs fajitas grapes )
echo "food: ${food[@]:1:3}"  # food: bananas cucumbers dates
echo "food: ${food[@]:4}"  # food: eggs fajitas grapes
echo "food: ${food[@]::3}"  # food: apples bananas cucumbers

# 数组的 增加 和 删除
food+=(liuxiang)
echo "food: ${food[@]}"  # food: apples bananas cucumbers dates eggs fajitas grapes liuxiang
unset food[-1]
echo "food: ${food[@]}"  # food: apples bananas cucumbers dates eggs fajitas grapes





