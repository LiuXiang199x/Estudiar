#include<iostream>

using namespace std;

// typedef为C语言的关键字，作用是为一种数据类型定义一个新名字。这里的数据类型包括内部数据类型（int,char等）和自定义的数据类型（struct等）。
// 在编程中使用typedef目的一般有两个，一个是给变量一个易记且意义明确的新名字，另一个是简化一些比较复杂的类型声明。

int main(int argc, char **argv){

	typedef int Status;  //指定标识符Status代表int类型
	typedef double DATE;  //指定标识符DATE代表double类型

	typedef int NUM[100]; //声明NUM为整数数组类型，可以包含100个元素

	typedef struct  //在struct之前用了关键字typedef，表示是声明新类型名
	{
		int month;
		int day;
		int year;  
	} TIME; //TIME是新类型名，但不是新类型，也不是结构体变量名
	// 上面写法和 struct TIME{}; 是一样的，只是写法区别

	int i; double j; // 等价于 Status i; DATE j;
	Status a; DATE b;
	TIME birthday;
	birthday.month = 10;
	birthday.day = 26;
	birthday.year = 1996;
	
	cout << "birthday:" <<	birthday.year << " " << birthday.month << " " << birthday.day << endl;

	NUM n;   //定义n为包含100个整数元素的数组，n就是数组名

	return 0;
}
