#include "struct_array.h"

using namespace std;


struct stu{
    char *name;  //姓名
    int num;  //学号
    int age;  //年龄
    char group;  //所在小组 
    float score;  //成绩
}classs[5];

int struct_array(){
    int i, num_140 = 0;
    float sum = 0;
    // struct stu class;
    for(i=0; i<5; i++){
        sum += classs[i].score;
        if(classs[i].score < 140) num_140++;
    }
    printf("sum=%.2f\naverage=%.2f\nnum_140=%d\n", sum, sum/5, num_140);

    return 0;
}