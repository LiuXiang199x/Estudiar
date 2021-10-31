class Animal:

    def eat(self):
        print("eat")

    def drink(self):
        print("drink")

    def sleep(self):
        print("sleep")

    def run(self):
        print("run")


class Dog(Animal):

    def bark(self):
        print("bark")


class XiaoTianQuan(Dog):

    def fly(self):
        print("fly")

    def bark(self):
        print("xiaotianquan fly")

# dog继承animal，xiaotianquan继承dog

xtq = XiaoTianQuan()
xtq.bark()   # “xiaotianquan fly”子类和父类一样时，子类会直接覆盖父类
