
var_:有时候，一个变量的最合适的名称已经被一个关键字所占用。 因此，像class或def这样的名称不能用作Python中的变量名称。 在这种情况下，你可以附加一个下划线来解决命名冲突
>>> def make_object(name, class):
SyntaxError: "invalid syntax"

>>> def make_object(name, class_):
...    pass


单前导下划线：_var:下划线前缀的含义是告知其他程序员：以单个下划线开头的变量或方法仅供内部使用。


双前导下划线 __var:名称修饰（name mangling） - 解释器更改变量的名称，以便在类被扩展的时候不容易产生冲突。
    class A:
        def __init__(self) -> None:
            self.a = 1
            self._a = 1
            self.__a = 1
    上方a，_a，__a都是类A的属性，三者都可以被继承但是a，_a都可以被子类修改，__a不行
    (自己尝试了，python3似乎子类可以完全继承和修改，但是双下划线是不能被访问的)



双前导和双末尾下划线 _var_: