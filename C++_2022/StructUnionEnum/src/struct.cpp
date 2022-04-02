#include "struct.h"

using namespace std;

// 单纯的一个结构体
struct stu
{
    char *name;  //姓名
    int num;  //学号
    int age;  //年龄
    char group;  //所在学习小组
    float score;  //成绩
};

// 定义结构体同时定义变量
struct stu1{
    char *name;  //姓名
    int num;  //学号
    int age;  //年龄
    char group;  //所在学习小组
    float score;  //成绩
} stu1, stu2;

// 如果只需要 stu1、stu2 两个变量，后面不需要再使用结构体名定义其他变量，那么在定义时也可以不给出结构体名
// struct{...} stu1, stu2;
struct{
    char *name;  //姓名
    int num;  //学号
    int age;  //年龄
    char *group;  //所在学习小组
    float score;  //成绩
} stu11, stu22 = {"Tom", 12, 16, "A", 145.2};

// 需要注意的是，结构体是一种自定义的数据类型，是创建变量的模板，不占用内存空间；
// 结构体变量才包含了实实在在的数据，需要内存空间来存储。

int struct_variable(){

    return 0;
}