#include<iostream>
#include<math.h>

using namespace std;

/*
最简单的钱币找零问题：这个问题在我们的日常生活中很普遍。假设1元、2元、5元、10元、20元、50元、100元的纸币分别有c0, c1, c2, c3, c4, c5, c6张。现在要用这些钱来支付K元，至少要用多少张纸币？用贪心算法的思想，很显然，每一步尽可能用面值大的纸币即可。在日常生活中我们自然而然也是这么做的。在程序中已经事先将Value按照从小到大的顺序排好。
思路：当前最好的选择，首先肯定是使用面值最大的钱，比如总共要130元，则第一步肯定是选择100元面值的，第二步选择20元面值的，第三步选择10元面值的。
*/

struct Money{
	int single[7] = {1,2,5,10,20,50,100};
	int number[7] = {2,5,0,3,4,0,4};
};

int total[7]={};

int changemoney(int money, Money money_param){

	cout << sizeof(money_param.single)/sizeof(money_param.single[0]) << endl;
	for(int i=sizeof(money_param.single)/sizeof(money_param.single[0])-1; i>=0; i--){
		total[i]=min(money_param.number[i], money/money_param.single[i]);
		money = money - total[i]*money_param.single[i];
	}
	return 0;
}


int main(int argc, char **argv){

	int money_needed;
	cout << "Please enter the amount of money you need change:";
	cin >> money_needed;
	
	Money money_param;
	
	// 这里就不考虑小数了
	if(money_needed<=0){
		cout << "WRONG NUMBER, Please enter again!" << endl;		
	}
	else{
		changemoney(money_needed, money_param);		
	}

	for(int i=0; i<sizeof(total)/sizeof(total[0]); i++){
		cout << "Amount of " << money_param.single[i];
		cout << ":" << total[i] << endl;
	}

	return 0;
}
