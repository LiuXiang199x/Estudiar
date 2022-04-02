#include "GetLine.h"

using namespace std;

int test_getline(){
    const char *a[] = {"aa", "qwe", "qweww", "13", "we", "eqwe"};

    ofstream outFile("../files/getlines.txt", ios::out);
    if(!outFile){
        cout << "error opening" << endl;
        return 0;
    }
    for(int i=0; i<6; i++){
        outFile << *a[i] << " " << a[i] << "\n";
    }
    outFile.close();

    char c[40];
    ifstream inFile("../files/getlines.txt", ios::in);
    if(!inFile){
        cout << "error opening" << endl;
        return 0;
    }

    //从 in.txt 文件中读取一行字符串，最多不超过 39 个
    cout << "======== 单行读取 ========" << endl;
    inFile.getline(c, 40);
    cout << c << endl;

    cout << "======== 多行读取 ========" << endl;
    while(inFile.getline(c, 40)){
        cout << c << endl;
    }
    inFile.close();

    return 0;
}
