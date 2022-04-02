#ifndef _TEST_H_
#define _TEST_H_
#include "allLibs.h"

class Adder{

    public:
        // 构造函数，每次默认初始化值为0
        Adder(int i=0);

        // 对外接口，添加需要加的值
        void addNum(int number);

        // 对外接口，获取最终结果
        int getTotal();
    
    private:
        // 内部数据，对外隐藏
        int total;
};

#endif // !_TEST_H_