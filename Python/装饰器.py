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
a = test2()
print(a)  # <function test2.<locals>.greet at 0x7fb81e97b0d0>
# print(a) 打印不出东西很正常，因为a是一个函数
print(a())
