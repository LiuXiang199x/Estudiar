from re import X
import torch
import torch.nn as nn

x = torch.rand(1, dtype=torch.float,requires_grad=True)
print(x.grad)

y = 4*x*x + 1
print(y.backward())
print(y.grad)
print(x.grad)

# backward()的理解：当前Variable(理解成函数Y)对leaf variable（理解成变量X=[x1,x2,x3]）求偏导。

w = torch.tensor([1.], requires_grad = True)
x = torch.tensor([2.], requires_grad = True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)
print(w, x, a, b, y)
# y.backward()
a.backward()
print(w.grad)

#查看叶子节点
print("is_leaf:\n",w.is_leaf,x.is_leaf,a.is_leaf,b.is_leaf,y.is_leaf)
#查看梯度
print("gradient:\n",w.grad,x.grad,a.grad,b.grad,y.grad)