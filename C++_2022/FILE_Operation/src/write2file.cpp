#include "write2file.h"

using namespace std;

int test() {
    printf("i am in write2file.h and write2file.cpp!!\n");
    const char *strrr = "wo hao shuai a !!!!!!!";
    //创建一个 fstream 类对象
    fstream fs;
    //将 test.txt 文件和 fs 文件流关联
    fs.open("/home/agent/C-_Project/C++_2022/FILE_Operation/files/test2.txt", ios::out);
    //向test.txt文件中写入 url 字符串
    cout << sizeof(strrr) << " " << sizeof(strrr[0]) << endl;   // 8, 1
    cout << strlen(strrr) << endl;
    fs.write(strrr, strlen(strrr));
    fs.close();
    return 0;
}


/*
char a[] = "aaaaa";
int length = sizeof(a)/sizeof(char); // length=6

不能这样做来获取 char * 的长度
char *a = new char[10];
int length = sizeof(a)/sizeof(char);

*/