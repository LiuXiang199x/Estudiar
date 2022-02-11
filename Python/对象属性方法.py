# Python一切皆对象，那么Python究竟是怎么管理对象的呢？
# __dict__
# 所有能通过PyObject类型的指针访问的都是对象。整数、字符串、元组、列表、字典、函数、模块、包，栈等都是对象。
# 所有对象都有三种特性: id(id())、类型（type())、值

# 对象：A 是类， a = A() 实例化了类A，创建了一个A对象a
# 类的属性（变量）：一般分为公有属性和私有属性，默认情况下所有得属性都是公有的，如果属性的名字以两个下划线开始，就表示为私有属性，没有下划线开始的表示公有属性。 python的属性分为实例属性和静态属性，实例属性是以self为前缀的属性，如果构造函数中定义的属性没有使用self作为前缀声明，则该变量只是普通的局部变量，类中其它方法定义的变量也只是局部变量，而非类的实例属性。
# 类的方法（函数）：类的方法也分为公有方法和私有方法，私有方法不能被模块外的类或者方法调用，也不能被外部的类或函数调用。python利用staticmethon或@staticmethon 修饰器把普通的函数转换为静态方法

class A(object):
    """
    Class A.
    """

    a = 0
    b = 1

    # __init__方法：构造函数用于初始化类的内部状态，为类的属性设置默认值（是可选的）。
    def __init__(self):
        self.a = 2   #实例（公有）属性，以self为前缀，可调用
        self.b = 3   #实例（公有）属性，以self为前缀，可调用
        self.__c = 4 #实例（私有）属性，以self为前缀，不可调用
        d = 5     #局部变量，不以self为前缀. 外部无法调用
        self._e = 6

    def test(self):
        print ('a normal func.')
        
    def printc(self):  #类方法
        print(self.__c)  # #打印出私有变量
         
    def __test2(self):   # 同理函数也是一样
        print("test2")

    def static_test(self):
        print ('a static func.')

    def class_test(self):
        print ('a calss func.')
        
print("type(A.__dict__): \n", type(A.__dict__))    
print("\ntype(A().__dict__): \n", type(A().__dict__))

print("\nA.__dict__: \n", A.__dict__)
print("\nA().__dict__: \n", A().__dict__)
print("由此可见， 类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的")
for name in A().__dict__:
    print(name)

print(A()._e)
print(A().test())
print("---")
print(A().printc())