#include<iostream>
#include<string>

using namespace std;

int main(int argc, char** argv){
	
	char c='1';
	int i1;
	cout << "char 和 int之间只能单个单个字符转换，并且是根据ASCII码转换的！" << endl;
	cout << "char 转 int —— int x = n - '0' ;   n为字符0~9其中一个。" << endl;
	cout << "int 转 char —— char x= n + '0' ;  n为数字0~9其中一个。" << endl;
	cout << "ASCII: 0-9 = 48-57" << endl;	
	cout << "a~z: 97~122的二进制编码" << endl;
	cout << "A~Z: 65~90的二进制编码" << endl;
	cout << "大写字母=小写字母-32 ；" << endl;
	i1 = c;
	cout << "char(1)的Ascii码为：" << i1 << endl;

	// 所以char ‘1’ 变成 int 1很简单: '1' - '0'
	cout << "int('1'-'0')=" << int('1'-'0') << endl;
	// 所以int 1 变成 char ‘1’也很简单: '1' + '0'	

	cout << "" << endl;

	// string2int
	string str="234";
	int n=stoi(str);  // 一步到位，可以选择不同进制
	cout << n << endl;
	// 也可以先把string转为字符指针。
	int nn=atoi(str.c_str());
	cout << nn << endl;

	// int 2 string
	int abc = 123;
	cout << typeid(to_string(abc)).name() << " " << to_string(abc) << endl;

	return 0;
}
