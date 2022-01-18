from ctypes import c_bool
from sys import api_version

print("Python的动态参数有两种，分别是*args和**kwargs，这里面的关键是一个和两个星号的区别，而不是args和kwargs在名字上的区别，实际上你可以使用*any或**whatever的方式。但就如self一样，默认大家都使用*args和**kwargs。")
"""
总结： args 和 kwargs必须连起来用！
"""

class argsss:
    
    def __init__(self):
        print("一个星号表示接收任意个参数。调用时，会将实际参数打包成一个元组传入形式参数。如果参数是个列表，会将整个列表当做一个参数传入。")
        
    def printt(a,b,c):
        print(a, b, c)
        
    def printt_(*l):
        # printt_(1,2,3)
        # print(*l)  # 1 2 3
        # print(l)   # (1, 2, 3)
        
        # printt_((1,2,3))
        # print(*l)  # (1, 2, 3)
        # print(l)   # ((1, 2, 3),)

        # printt_(*(1,2,3))
        # print(*l)  # 1, 2, 3
        # print(l)   # (1, 2, 3)
        
        # printt_([1,2,3])
        print(*l)  # [1, 2, 3]
        print(l)   # [1, 2, 3],)

        # printt_(*[1,2,3])
        # print(*l)  # 1, 2, 3
        # print(l)   # (1, 2, 3)
                
    def printtt(x, y, *a):
        # print(x, y, *a)   # 1 2 3 4 5 6
        # print(x, y, a)   # 1 2 (3 4 5 6)
        print("x:",x)
        print("y:",y)
        print("a:",a)
        print("*a:",*a)
        # 如果输入数据不够，比如只有

    def prin(x1, x, y, *args):
        # print(x, y, *a)   # 1 2 3 4 5 6
        # print(x, y, a)   # 1 2 (3 4 5 6)
        print("x:",x)
        print("y:",y)
        print("args:",args)
        print("*args:",*args)
        # 如果输入数据不够，比如只有

class kwargsss:
    def __init__(self):
        print(" **kwargs 打包关键字参数成dict给函数体调用")

    def print_(*args, **kwargs):
        print(kwargs)

def func(*args):
    """
    *表示接收任意个数量的参数，调用时会将实际参数打包为一个元组传入实参
    :param args:
    :return:
    """
    print(args)

def foo(*args, **kwargs):
	print ('args = ', args)
	print ('kwargs = ', kwargs)
	print ('---------------------------------------')
'''
    可以看到，这两个是python中的可变参数。
    args表示任何多个无名参数，它是一个tuple；
    **kwargs表示关键字参数，它是一个 dict。
    并且同时使用args和kwargs时，必须*args参数列要在kwargs前，
    像foo(a=1, b=‘2’, c=3, a’, 1, None, )这样调用的话，
    会提示语法错误“SyntaxError: non-keyword arg after keyword arg”。 
'''

if __name__ == "__main__":
    func(123, 'hello', ['a', 'b', 'c'], {'name': 'kobe', 'age': 41})
    foo(1,2,3,4)
    foo(a=1,b=2,c=3)
    foo(1,2,3,4, a=1,b=2,c=3)
    foo('a', 1, None, a=1, b='2', c=3)
    
    A = kwargsss()
    A.print_(a=1,b=2)