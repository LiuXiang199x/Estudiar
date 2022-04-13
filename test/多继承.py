class A:

    def __init__(self):
        self.name = "AAA"

    def demo(self):
        print("A demo 方法")

    def test(self):
        print("A test 方法")


class B:
   
    def __init__(self):
        self.name = "BBB"
 
    def demo(self):
        print("B demo 方法")

    def test(self):
        print("Atest 方法")


class C(A, B):


    pass

# yj
c = C()
print(c.name)
c.demo()
c.test()

# 确定C类的调用方法
# (<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class 'object'>)
# 首先查看方法是不是在C类中，如果不是继续向下查看是否在A类中，如果还是没有便继续向下查找，如果有的话就停止查找执行找到的方法。
print(C.__mro__)
