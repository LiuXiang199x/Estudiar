# 类__dict__属性：静态函数，类函数，普通函数 | 全局变量，内置属性
# 类对象__dict__属性：储存了self.xx一些东西，父类子类对象功用__dict__
# 查看对象的属性
from 对象属性方法 import A

print("===========================")
for name in A.__dict__:
    print(name)
print("--------------------")
for name in A().__dict__:
    print(name)