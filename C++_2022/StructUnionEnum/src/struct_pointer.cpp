#include "struct_pointer.h"

using namespace std;


//结构体
struct stu{
    const char *name;  //姓名
    int num;  //学号
    int age;  //年龄
    char group;  //所在小组
    float score;  //成绩
} stu10 = { "Tom", 12, 18, 'A', 136.5 }, *pstr2=&stu10,
stus[] = {
    {"Li ping", 5, 18, 'C', 145.0},
    {"Zhang ping", 4, 19, 'A', 130.5},
    {"He fang", 1, 18, 'A', 148.5},
    {"Cheng ling", 2, 17, 'F', 139.0},
    {"Wang ming", 3, 17, 'B', 144.5}
};


int struct_pointer(){
    struct stu *pstr = &stu10;

    // const char* pstt = { "Tom", "12", "18", 'A', "136.5" };
    // char psts[5] = { "Tom", "12", "18", 'A', "136.5" };
    int len = sizeof(stu10) / sizeof(struct stu);
    cout << sizeof(stu10) << " " << sizeof(struct stu) << endl;  // 24, 24

    cout << &stu10 << endl;
    cout << stu10.age << endl;
    // .的优先级高于*，(*pointer)两边的括号不能少
    // 如果去掉括号写作*pointer.memberName，那么就等效于*(pointer.memberName)，这样意义就完全不对了
    cout << (*pstr).age << endl;
    // ->是一个新的运算符，习惯称它为“箭头”，有了它，可以通过结构体指针直接取得结构体成员；这也是->在C语言中的唯一用途。
    cout << pstr2->group << endl;

    cout << len << endl;

    return 0;
}

void average(struct stu *ps, int len){
    int i, num_140 = 0;
    float average, sum = 0;
    for(i=0; i<len; i++){
        sum += (ps + i) -> score;
        if((ps + i)->score < 140) num_140++;
    }
    printf("sum=%.2f\naverage=%.2f\nnum_140=%d\n", sum, sum/5, num_140);
}

int struct_FuncParams(){

    int len = sizeof(stus) / sizeof(stu);
    cout << len << endl;
    average(stus, len);

    return 0;
}
