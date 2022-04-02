#include "ReadWriteBinaryf.h"

using namespace std;

class CStudent
{
    public:
    char szName[20];  //假设学生姓名不超过19个字符，以 '\0' 结尾
    // char szId[0];  //假设学号为9位，以 '\0' 结尾
    int age;  //年龄
};


int write2binary(){

    CStudent s;
    ofstream outFile("../files/BinaryStudent.dat", ios::binary | ios::out);

    while (cin >> s.szName >> s.age)
        outFile.write((char*)&s, sizeof(s));
    outFile.close();
    
    return 0;
}

int readfrombinary(){

    CStudent ss;
    ifstream inFile("../files/BinaryStudent.dat", ios::binary | ios::in);
    if(!inFile){
        cout << "error openining dat file" << endl;
        return 0;
    }
    while (inFile.read((char *)&ss, sizeof(ss))){
        cout << ss.szName << " " << ss.age << endl;
    }
    inFile.close();

    return 0;
}