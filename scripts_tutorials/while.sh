#!/usr/bin/env bash

# 写在一行就要加分号隔开
echo -n "enter a true/false to activate while loop:"
read a
while $a; do
	echo "this is a loop while";
done


# until循环与while循环恰好相反，只要不符合判断条件（判断条件失败），就不断循环执行指定的语句。一旦符合判断条件，就退出循环。
echo -n "enter a true/false to activate until loop:"
read a
until $a; do echo "hi, until looping...."; done

for i in a "1 2 b" c;do
	echo $i
done

# C语言的用法
for (( i=0; i<5; i=i+1 )) do
	echo $i
done

# break and continue
for number in 1 2 3 4 5 6
do
	echo $number
	if [ $number = "4" ]; then  # 一定要用双引号！！！不能用单引号！！
		break
	fi
done

