#include "Union.h"

using namespace std;

struct data2{
    int n;
    char ch;
    short m;
    short ww;
    double a;
};

union data{
    int n;
    char ch;
    short m;
};

int test_union(){
    union data a;

    // char: 1 int: 4 float: 4 short:2 double: 8
    cout << "char: " << sizeof(char) << " int: " << sizeof(int) << \
    " float: " << sizeof(float) << " short:" << sizeof(short) << " double: " << sizeof(double) << endl;

    cout << "size of data: " << sizeof(data) << endl;  // 4
    cout << "size of data2: " << sizeof(data2) << endl;  // 4

    printf("%d, %d\n", sizeof(a), sizeof(union data) );
    a.n = 0x40;
    printf("%X, %c, %hX\n", a.n, a.ch, a.m);
    a.ch = '9';
    printf("%X, %c, %hX\n", a.n, a.ch, a.m);
    a.m = 0x2059;
    printf("%X, %c, %hX\n", a.n, a.ch, a.m);
    a.n = 0x3E25AD54;
    printf("%X, %c, %hX\n", a.n, a.ch, a.m);
   
    return 0;
}
