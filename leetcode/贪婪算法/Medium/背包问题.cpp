#include<iostream>

using namespace std;

/*
有一个背包，最多能承载重量为 C=150的物品，现在有7个物品（物品不能分割成任意大小），编号为 1~7，重量分别是 wi=[35,30,60,50,40,10,25]，价值分别是 pi=[10,40,30,50,35,40,30]，现在从这 7 个物品中选择一个或多个装入背包，要求在物品总重量不超过 C 的前提下，所装入的物品总价值最高。
*/
/*
里需要明确的几个点：1.每个物品都有重量和价值两个属性；2.每个物品分被选中和不被选中两个状态（后面还有个问题，待讨论）；3.可选物品列表已知，背包总的承重量一定。所以，构建描述每个物品的数据体结构 OBJECT和背包问题定义为：//typedef是类型定义的意思
*/


// 定义待选物体的结构体类型
typedef struct tagObject{
	int weight;
	int price;
	int status;
};


// 定义背包问题
typedef struct tagKnapsackProblem{
	vector<OBJECT>objs;
	int totalC;
};

/*
策略1：价值主导选择，每次都选价值最高的物品放进背包；

策略2：重量主导选择，每次都选择重量最轻的物品放进背包；

策略3：价值密度主导选择，每次选择都选价值/重量最高的物品放进背包。
*/

// 策略1：价值主导选择：每次都选价值最高的物品放进背包根据这个策略最终选择装入背包的物品编号依次是 4、2、6、5，此时包中物品总重量是 130，总价值是 165。
//遍历没有被选的objs,并且选择price最大的物品,返回被选物品的编号
int Choosefunc1(std::vector<OBJECT>& objs, int c)
{
    int index = -1;  //-1表示背包容量已满
    int max_price = 0;
    //在objs[i].status == 0的物品里，遍历挑选objs[i].price最大的物品
    for (int i = 0; i < static_cast<int>(objs.size()); i++)
    {
        if ((objs[i].status == 0) && (objs[i].price > max_price ))//objs没有被选,并且price> max_price
        {
            max_price  = objs[i].price;
            index = i;
        }
    }
    return index;
}

// 策略2：重量主导选择: 每次都选择重量最轻(小)的物品放进背包根据这个策略最终选择装入背包的物品编号依次是 6、7、2、1、5，此时包中物品总重量是 140，总价值是 155。
int Choosefunc2(std::vector<OBJECT>&objs, int c)
{
    int index = -1;
    int min_weight= 10000;
    for (int i = 0; i < static_cast<int>(objs.size()); i++)
    {
        if ((objs[i].status == 0) && (objs[i].weight < min_weight))
        {
            min_weight= objs[i].weight;
            index = i;
        }
    }
    return index;
}


// 策略3：价值密度主导选择：每次选择都选价值/重量最高(大)的物品放进背包物品的价值密度 si 定义为 pi/wi，这 7 件物品的价值密度分别为 si=[0.286,1.333,0.5,1.0,0.875,4.0,1.2]。根据这个策略最终选择装入背包的物品编号依次是 6、2、7、4、1，此时包中物品的总重量是 150，总价值是 170。

int Choosefunc3(std::vector<OBJECT>& objs, int c)
{
    int index = -1;
    double max_s = 0.0;
    for (int i = 0; i < static_cast<int>(objs.size()); i++)
    {
        if (objs[i].status == 0)
        {
            double si = objs[i].price;
            si = si / objs[i].weight;
            if (si > max_s)
            {
                max_s = si;
                index = i;
            }
        }
    }
    return index;
}

void GreedyAlgo(KNAPSACK_PROBLEM *problem, SELECT_POLICY spFunc)
{
    int idx;
    int sum_weight_current = 0;
    //先选
    while ((idx = spFunc(problem->objs, problem->totalC- sum_weight_current)) != -1)
    {   //再检查，是否能装进去
        if ((sum_weight_current + problem->objs[idx].weight) <= problem->totalC)
        {
            problem->objs[idx].status = 1;//如果背包没有装满，还可以再装,标记下装进去的物品状态为1
            sum_weight_current += problem->objs[idx].weight;//把这个idx的物体的重量装进去，计算当前的重量
        }
        else
        {
            //不能选这个物品了，做个标记2后重新选剩下的
            problem->objs[idx].status = 2;
        }
    }
    PrintResult(problem->objs);//输出函数的定义，查看源代码
}

OBJECT objects[] = { { 35,10,0 },{ 30,40,0 },{ 60,30,0 },{ 50,50,0 }, { 40,35,0 },{ 10,40,0 },{ 25,30,0 } };             
    
int main()
{
    KNAPSACK_PROBLEM problem;
    problem.objs.assign(objects, objects + 7);//assign赋值，std::vector::assign
    problem.totalC = 150;
    cout << "Start to find the best way ,NOW" << endl;
    GreedyAlgo(&problem, Choosefunc3);
    system("pause");
    return 0;
}
