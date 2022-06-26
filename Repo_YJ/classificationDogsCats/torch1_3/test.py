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
y.backward()
print(w.grad)