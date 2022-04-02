#include "enum.h"

using namespace std;

// #define Mon 1
// #define Tues 2
// #define Wed 3
// #define Thurs 4
// #define Fri 5
// #define Sat 6
// #define Sun 7

// int enum_define(){

//     int day;
//     scanf("%d", &day);
//     switch(day){
//         case Mon: puts("Monday"); break;
//         case Tues: puts("Tuesday"); break;
//         case Wed: puts("Wednesday"); break;
//         case Thurs: puts("Thursday"); break;
//         case Fri: puts("Friday"); break;
//         case Sat: puts("Saturday"); break;
//         case Sun: puts("Sunday"); break;
//         default: puts("Error!");
//     }

//     return 0;
// }


int enum_demo(){

    // 枚举值默认从 0 开始，往后逐个加 1（递增）；也就是说，week 中的 Mon、Tues ...... Sun 对应的值分别为 0、1 ...... 6。
    // 输入4，5，6还是打印不出 Jueves等是因为这几个数值已经被 宏定义define占掉了
    enum week{Lunes, Martes=12, Miercoles, Jueves, Viernes, Sabado, Domingo} day;
    
    // 0 Z9enum_demovE4week;  1 Z9enum_demovE4week
    cout << Lunes << " " << typeid(Lunes).name() << endl;
    cout << Martes << " " << typeid(Martes).name() << endl;
    cout << Miercoles << " " << typeid(Miercoles).name() << endl;
    cout << day << " " << typeid(day).name() << endl;
    
    scanf("%d", &day);
    switch(day){
        case Lunes: puts("Lunes"); break;
        case Martes: puts("Martes"); break;
        case Miercoles: puts("Miercoles"); break;
        case Jueves: puts("Jueves"); break;
        case Viernes: puts("Viernes"); break;
        case Sabado: puts("Sabado"); break;
        case Domingo: puts("Domingo"); break;
        default: puts("Error!");
    }

    return 0;
}