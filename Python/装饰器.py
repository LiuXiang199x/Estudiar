"""
装饰器(Decorators)是 Python 的一个重要部分。
简单地说：他们是修改其他函数的功能的函数。他们有助于让我们的代码更简短。
"""

def test1():
    print("函数作为变量/删除函数名 | 函数内部嵌套函数")
    def hi(name="yasoob"):
        return "hi " + name
    
    print(hi())
    # output: 'hi yasoob'


    # 我们甚至可以将一个函数赋值给一个变量，比如
    greet = hi
    # 我们这里没有在使用小括号，因为我们并不是在调用hi函数
    # 而是在将它放在greet变量里头。我们尝试运行下这个
    
    print(greet())   # output: 'hi yasoob'
    
    # # 如果我们删掉旧的hi函数，看看会发生什么！
    # del hi
    # print(hi())
    # #outputs: NameError

    print(greet())  #outputs: 'hi yasoob'

def test2(name = "ni da ye"):
    print("函数内部返回函数")
    def greet():
        return "now you are in the greet() function"

    def welcome():
        return "now you are in the welcome() function"
 
    if name == "ni da ye":
        return greet
    else:
        return welcome


def decorator_test1():
    print("decorator它们封装一个函数，并且用这样或者那样的方式来修改它的行为。")

    def funA(desA):
        print("It's funA")
    
    def funB(desB):
        print("It's funB")

    # 只能放在函数前，所以被他修饰的一定是在他后面的函数
    @funA
    def funC():
        print("It's funC")
    """
    @funA 修饰函数定义def funC()，将funC()赋值给funA()的形参。
    执行的时候由上而下，先定义funA、funB，然后运行funA(funC())。
    此时desA=funC()，然后funA()输出‘It's funA'。
    """

    # funB(funA(funD()))
    @funB
    @funA
    def funD():
        print("It's funD")
    # @funB 修饰装饰器@funA，@funA 修饰函数定义def funD()，
    # 将funD()赋值给funA()的形参，再将funA(funD())赋值给funB()。
    # 执行的时候由上而下，先定义funA、funB，然后运行funB(funA(funD()))。
    # 此时desA=funD()，然后funA()输出‘It's funA'；
    # desB=funA(funD())，然后funB()输出‘It's funB'。


def decorator_test2():
    def funA(desA):
        print("It's funA")

        print('---')
        print(desA)
        desA()
        print('---')

    def funB(desB):
        print("It's funB")

    @funB
    @funA
    
    def funC():
        print("It's funC")

def decorator_test3():
    def funA(desA):
        print("It's funA")

    def funB(desB):
        print("It's funB")
        print('---')
        print(desB)

    @funB
    @funA
    def funC():
        print("It's funC")
        
# 上面将funC()作为参数传给funA，那么funA(funC())怎么传给funB()呢？
# 打印desB，发现并没有参数传递。
# 是否可以理解为当‘装饰器' 修饰 ‘装饰器'时，仅是调用函数。