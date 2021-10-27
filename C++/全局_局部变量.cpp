#include<iostream>

using namespace std;

int b = 0;

void func(int i){
	b=1;
	i = 0;
}

int main(int argc, char** argv){

	int i;
	i = 9;
	cout << "in main loop:" << i << endl;  // 9
	cout << b << endl;  // 0
	func(i);
	cout << i << endl;  // 9
	cout << b << endl;  // 1

	return 0;
}
