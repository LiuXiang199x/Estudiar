#include"alllib.h"

namespace everest
{
    namespace planner
    {
        class RlApi {
        public:
            //bool processTarget(const int& idx, const int& idy, int& res_idx, int& res_idy);
            bool processTarget(const int& idx, const int& idy, int& res_idx, int& res_idy) {
                cout << "Im in processTarget" << "idx=" << idx << "idy=" << idy << endl;
                cout << "res_idx=" << res_idx << "res_idy=" << res_idy << endl;
                res_idx = 12;
                res_idy = 22;
                // int a = 0;
                // int b = 0;
                processTarget2(res_idx, res_idy);
                // cout << a << " " << b << endl;
                // res_idx = a;
                // res_idy = b;

                cout << "Im in terminate processTarget" << endl;
                return true;
            }

            void processTarget2(int& res_idx, int& res_idy) {
                printf(" i am in processTarget22222");
                printf(" i am in processTarget212121");
                res_idx = 1234;
                res_idy = 1242;
            }
        };
    }
}

