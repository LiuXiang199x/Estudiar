#include "pointerRW.h"

using namespace std;


class CSstudent{
    public:
    char szName[20];
    int age;
};


int test_pointer(){

    CSstudent s;       
    fstream ioFile("../files/BinaryStudent.dat", ios::in|ios::out);//用既读又写的方式打开
    if(!ioFile) {
        cout << "error" ;
        return 0;
    }
    ioFile.seekg(0,ios::end); //定位读指针到文件尾部，
                              //以便用以后tellg 获取文件长度
    int L = 0,R; // L是折半查找范围内第一个记录的序号
                  // R是折半查找范围内最后一个记录的序号
    R = ioFile.tellg() / sizeof(CSstudent) - 1;
    //首次查找范围的最后一个记录的序号就是: 记录总数- 1
    do {
        int mid = (L + R)/2; //要用查找范围正中的记录和待查找的名字比对
        ioFile.seekg(mid *sizeof(CSstudent),ios::beg); //定位到正中的记录
        ioFile.read((char *)&s, sizeof(s));
        int tmp = strcmp( s.szName,"Jack");
        if(tmp == 0) { //找到了
            s.age = 20;
            ioFile.seekp(mid*sizeof(CSstudent),ios::beg);
            ioFile.write((char*)&s, sizeof(s));
            break;
        }
        else if (tmp > 0) //继续到前一半查找
            R = mid - 1 ;
        else  //继续到后一半查找
            L = mid + 1;
    }while(L <= R);
    ioFile.close();

    return 0;
}
