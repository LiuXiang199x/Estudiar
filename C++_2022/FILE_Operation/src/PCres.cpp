#include "PCres.h"

using namespace std;
using namespace everest::ai;

PCrest::PCrest(int aaa){
    cout << "Start init params =======>>>>>> " << endl;
    init_Params();
    nano_res[1][0] = 111;
    nano_res[1][1] = 222;
    papa = aaa;
}

int PCrest::get_papa(){
    return papa;
}

void PCrest::set_papa(int num){
    papa = num;
}

PCrest::~PCrest(){}

void PCrest::init_Params(){
    cout << "im in init_Params" << endl;
    // nano_res[2][2] = {1, 2, 3, 4};     // 该方法不行
    // nano_res[2][2] << 1, 2, 3, 4;      // 该方法不行
    nano_res[0][0] = 1;
    nano_res[0][1] = 2;

    pics_url[0] = "123";
    pics_url[1] = "222";
}

int PCrest::test(){
    everest::ai::PCrest qq(24);
    cout << qq.pics_url[0] << endl;
    cout << qq.pics_url[1] << endl;
    for(int i=0; i<2; i++){
        cout << qq.nano_res[0][i] << " ";
        cout << qq.nano_res[1][i] << " ";
    }
    cout << endl;

    cout << qq.get_papa() << endl;
    qq.set_papa(12345);
    cout << qq.get_papa() << endl;
    return 0;
}