#include "<<_>>write2file.h"

using namespace std;

int test1(){
    printf("fstream或者ifstream类负责实现对文件的读取，它们内部都对>>输出流运算符做了重载；\
    同样，fstream和ofstream类负责实现对文件的写入，它们的内部也都对<<输出流运算符做了重载。\n");

    int x, sum = 0;
    // fstream gs;
    // gs.open("../files/test3.txt", ios::out);
    ifstream srcFile("../files/test3.txt", ios::in);

    // cout << typeid(srcFile).name() << endl;    // 存在返回1，不存在返回0
    // cout << !srcFile << endl;   // 1

    if (!srcFile){
        cout << "error opening source file." << endl;
        return 0;
    }

    // 以文本模式打开test3.txt备写
    ofstream destFile("../files/test4.txt", ios::out);
    if(!destFile){
        srcFile.close();
        cout << "error opening destination file." << endl;
        return 0;
    }

    // 可以像用cin那样用ifstream对象
    while (srcFile >> x){
        sum += x;
        // 可以像 cout 那样使用 ofstream 对象
        destFile << x << " ";
    }

    cout << "sum: " << sum << endl;
    destFile.close();
    srcFile.close();

    return 0;
}