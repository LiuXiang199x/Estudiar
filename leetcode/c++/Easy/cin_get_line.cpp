#include<iostream>
#include"stdio.h"
#include<string>

using namespace std;
void func_cin();
// cin ---- get() ----  getline()

void char_cin(){
	char input[5]; 
	char input2[10];
	for (int i=0; i<5; i++){
		std::cout << "inputs:";
		std::cin >> input[i];
	}
	std::cout << "inputs2:";
	std::cin >> input2;
	
	std::cout << std::endl;

	for (int i=0; i<5; i++){
		std::cout << input[i] << "-";
	}
	std::cout << std::endl;

	for (int i=0; i<10; i++){
		std::cout << input2[i] << "-";
	}
	std::cout << std::endl;
/*
cin>>会自动过滤掉不可见字符（如空格 回车 tab等）。若不想过滤掉空白字符，可以用noskipws流进行控制。
inputs:QD 12 Q
inputs:inputs:inputs:inputs:inputs2:QWE 21 2E

Q-D-1-2-Q-
Q-W-E--G-b-D-V---
*/
}

void cs_get(){
// 1）cin.get(字符变量名)，用来接收字符，只获取一个字符，可以接收空格，遇回车结束
// 2）cin.get(数组名，接收字符数目)，用来接收字符串，可以接收空格，遇回车结束。
// 3）cin.get()，没有参数，主要用于舍弃输入流中不需要的字符，或者舍弃回车，即舍弃输入流中的一个字符。
	char input[5];
	
	for(int i=0; i<5; i++){
		cin.get(input[i]);
		cout << input[i];
	}
	cout << "Condition1 over" << endl;
	cout << endl;

	cout << "cin.get(name, len)" << endl;
	char input2[10];
	cin.get(input2, 10);
	cout << input2;


}

int main(){
	// char_cin();
	cs_get();
	return 0;
}
