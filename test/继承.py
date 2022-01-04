#!/usr/bin/env python

class Person():

	def __init__(self, name, age):
		self.name = name
		self.age = age
		self.weight = "weight"

	def talk(self):
		print("You are in class Person")
		print(self.name)
		print(self.age)
		print(self.weight)

class Person2():
	def __init__(self, name, age):
		self.name = name
		self.age = age
	def talk(self):
		print("I am Person2!")
		print(self.name)
		print(self.age)

class Chinese(Person):  # 子类chinese继承父类Person

	def __init__(self, name, age, language):
		Person.__init__(self, name, age)
		self.language = language


	def walk(self):
		print("is walking...")
		print(self.name)
		print(self.age)
		print(self.language)
# 如果子类要对父类的方法进行重写，再想调用父类的方法，就需要使用super().方法名() 的形式
class Xiang(Person):
	pass

class Xiang2(Person, Person2):
	pass

c = Chinese('bigberg', 22, 'chinese')


d = Xiang(2,3)
d.talk()

x = Xiang2(2,3)
x.talk()

