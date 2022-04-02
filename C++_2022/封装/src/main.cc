#include "encapsulation.h"

using namespace std;

// 程序的主函数
int main( )
{
   // 你不给我值，我自己默认初始化位0
   Adder qqq;
   
   cout << qqq.getTotal() << endl;
   qqq.addNum(10);
   cout << qqq.getTotal() << endl;
   qqq.addNum(15);
   cout << qqq.getTotal() << endl;

   // 你给我值，我就初始化位你给的
   Adder qq(3);
   
   cout << qq.getTotal() << endl;
   qq.addNum(10);
   cout << qq.getTotal() << endl;
   qq.addNum(15);
   cout << qq.getTotal() << endl;

   return 0;
}