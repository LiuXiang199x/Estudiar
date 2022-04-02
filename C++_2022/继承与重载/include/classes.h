#ifndef _CLASSES_H_
#define _CLASSES_H_
#include "allLibs.h"

using namespace std;
// 被继承的类是 -> 基类；   继承的类是 -> 派生类
// 基类
class Shape 
{
   public:
      void setWidth(int w);
      void setHeight(int h);
      int pubbbb = 101010;
   
   protected:
      int width;
      int height;
      int proooo = 111000;
   
   private:
      int priiii = 100000;
};
 
class Shape2{
   public:
   int ubpp = 22222;
};

// 派生类， 单继承和多继承
class Rectangle: public Shape, public Shape2
{
   public:
      int getArea();
};

#endif // !_CLASSES_H_
