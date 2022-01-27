# ->
# @

# ->常常出现在python函数定义的函数名后面，
# 为函数添加元数据,描述函数的返回类型，从而方便开发人员使用。
# 这里返回的仍然是int类型， ->float只其一个注释作用
def add(x, y) -> float:   
    return x+y  # 返回的永远之只取决于我们输入的数

########################## @@@@@@@ #####################
def decorator(func):
       return func
 
@decorator
def some_func():
    pass

# 以上两个写法是一样的

