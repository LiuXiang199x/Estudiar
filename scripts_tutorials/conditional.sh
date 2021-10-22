#!/usr/bin/env bash

echo -n "enter a number:"
read a
if test $a = '1'; then echo 1;
elif test $a = '2'; then echo 2;
elif [ $a = '3' ]; then echo 3;    # 必须有空格！！括号前后要空格！！等号前后也要空格！！
else echo "it is not 1,2,3.!"
fi

if test $USER = 'xiang';then
	echo 'hello xiang!'
else
	echo "you are not xiang!"
fi

if [ -a '1.py' ]; then echo 'true'
elif [ -a 'demo1.sh' ]; then echo '2true'   # 会返回2true
fi

# 可以用 test 也可以用 (()) 来作为if的判断
if ((3>2)); then echo "3>2"; else echo "2<3"; fi

# 算术判断要用括号 （）
if ((1)); then echo "true 1"; fi
if ((0)); then echo "true 0"; else echo "false0"; fi

# (()) 也可以作为赋值语句
if (( foo = 5 )); then echo "foo is $foo"; fi
echo -n "输入一个1到3之间的数字（包含两端）> "
read character
case $character in
	1) echo 1;;
	2) echo 2;;
	3) echo 3;;
	"a") echo 4;;
	*) echo "i dont know your input";;
esac
