#include<iostream>

using namespace std;

// input:"{}", output: true;  input:"()[]",output: true;  input: "{]" output:false
// input:"{[]}", output: true;  input:"{[)}", output: false;  
// best way: time complex(O(N)), space complex(O(1))

int main(){
	char input[] = {"[]{}()"};
	for(int i=0; i<sizeof(input)/sizeof(input[0]); i++){

		// 这里前后加判断条件和ASCII表对应肯定是可以的
		if(input[i] == input[sizeof(input)/sizeof(input[0])-1]){
			cout << "true 1" << endl;	
		}
		else if(i%2!=0 && input[i]==input[i+1]){
			cout << "true 2" << endl;	
		}
		else{
			cout << "false" << endl;
		}

		cout << input[i] << " ";
	}

	return 0;
}

// 推荐做法：
class Solution{
public:
	bool isValid(string s){
		int top = -1;
		for(int i=0; i<s.length(); ++i){
			if(top<0 || !isMatch(s[top], s[i])){
				++top;
				s[top] = s[i];		
			}else{
				--top;		
			}
		}
		return top == -1;
		
	}
	bool isMatch(char c1, char c2){
		if(c1 == '(' && c2 == ")") return true;
		if(c1 == '{' && c2 == "}") return true;
		if(c1 == '[  ' && c2 == "]") return true;
		return false;	
	}
};
