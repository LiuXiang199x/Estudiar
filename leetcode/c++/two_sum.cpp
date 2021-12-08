#include<iostream>
#include<vector>

using namespace std;

// given number = [2,7,11,15], target = 9
// return [0, 1]

vector<int> get_input(){
	int n;
	cout << "please input lenght of array:";
	cin >> n;

	vector<int> inputs(n);	
	
	for(int i=0; i<n; i++){
		cout << "given_array[" << i << "]=";
		cin >> inputs[i];
		cout << endl;
	}

	cout << "given_array = ";
	for(int i=0; i<n; i++){
		cout << inputs[i] << " ";
	}
	cout << endl;
	return inputs;	
}

vector<int> solution_1(vector<int> arrays, int target){

	vector<int> result(2);
	for(int i=0; i<arrays.size(); i++){
		for(int j=arrays.size()-1; j>=i; j--){
			if(arrays[i]+arrays[j]==target){
				result[0] = i;
				result[1] = j;
				return result;
			}		
		}	
	}

}

// 空间换时间，借用Map多开点内存来存放记录，节省了时间
// 借助Map结构将数组中每个元素及其索引相互对应
vector<int> solution_best(vector<int> arrays, int target){

	vector<int> result(2);
	for(int i=0; i<arrays.size(); i++){
		int a = target - arrays[i];
		if (a>=0){
			class Solution {
public:
vector<int> twoSum(vector<int>& A, int target) {
unordered_map<int, int> m;
for (int i = 0; i < A.size(); ++i) {
int t = target - A[i];
if (m.count(t)) return { m[t], i };
m[A[i]] = i;
}
return {};
}
};
		}
	}

}

int main(){
	//最好的办法时间复杂度是O(n)
	int target;	
	vector<int> given_array;	
	cout << "please input the target number:";
	cin >> target;
	given_array = get_input();

	for(int i=0; i<given_array.size(); i++){
		cout << given_array[i] << " ";
	}
	cout << endl;

	vector<int> result1(2);
	vector<int> result2(2);
	result1 = solution_1(given_array, target);
	for(int i=0; i<result1.size(); i++){
		cout << "result: "<<result1[i]<<endl;	
	}

	result2 = solution_best(given_array, target);
	for(int i=0; i<result2.size(); i++){
		cout << "result: "<<result2[i]<<endl;	
	}
	return 0;
}



