#!/usr/bin/env python
# super() 函数是用于调用父类(超类)的一个方法。
# super() 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。
'''
如果子类继承了父类 子类重写了父类的同名方法 而且子类想调用父类的这个同名方法一共有三种方式
1 父类名.父类方法名(self)
2super(子类名, self).父类方法名()
3 super().父类方法名()
不要一说到 super 就想到父类！super 指的是 MRO 中的下一个类！
按继承顺序从下到上搜索，在括号内从左到右，MRO不能存在上下和左右关系，不然报错
C类继承与A和B，那在C类中调用super只能调用A类的函数，a里再用super调b的
'''
# Python3和Python2的一个区别是:Python3可以使用直接使用super().xxx代替super(Class, self).xxx
class Root():
	def __init__(self):
		print("this is root")


class B(Root):
	def __init__(self):
		print("enter B")
		super(B, self).__init__()
		print("leave B")

class C(Root):
	def __init__(self):
		print("enter C")	
		super(C, self).__init__()
		print("leave C")

class D(B, C):
	pass

d = D()
print(d.__class__.__mro__)
# enter B, enter C, this is root, leave C, leave B
# (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.Root'>, <class 'object'>)



