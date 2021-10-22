#!/usr/bin/env bash

echo -n "enter a number"
read a
if ((a==2));then
	echo "buena suerte!"
else
	echo "seguimos"
fi

rand(){
	min=$1
	max=$(($2-$min+1))
	num=$(($RANDOM+1000000000))
	echo $(($num%$max+$min))
}

world_array=(10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10)

for i in {1..100};do
	rnd=$(rand 1 16)
	while ((${world_array[$(($rnd-1))]}<=0));do
		rnd=$(rand 1 16)
	done
	
	if (($rnd==1));then
		echo "answer is $rnd"

	elif (($rnd==2));then
		echo "answer is $rnd"

	elif (($rnd==3));then
		echo "answer is $rnd"

	elif (($rnd==4));then
		echo "answer is $rnd"
	fi
done
