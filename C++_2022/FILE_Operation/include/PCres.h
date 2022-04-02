// 带参数的构造函数-析构函数
// 构造函数中初始化值,修改值等等
#ifndef _PCres_H_
#define _PCres_H_
#include <iostream>
#include <string>
#include "string.h"
#include <fstream>

namespace everest{
    namespace ai
    {
        class PCrest{
            public:
            int nano_res[2][2];   // 初始化不赋值的话会乱指
            std::string pics_url[2];
            void set_papa(int num);
            int get_papa();

            PCrest(int papa);
            ~PCrest();

            int test();

            private:
            void init_Params();
            int papa;

        };

    } // namespace ai
    
}

#endif // !_PCres_H_