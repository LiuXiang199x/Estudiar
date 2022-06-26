from re import X
import torch
import torch.nn as nn

x = torch.rand(1, dtype=torch.float,requires_grad=True)
print(x.grad)

y = 4*x*x*x + 2
print(y.backward())
print(y.grad)
print(x.grad)

# backward()的理解：当前Variable(理解成函数Y)对leaf variable（理解成变量X=[x1,x2,x3]）求偏导。

a=torch.ones(2,1,requires_grad=True)
b=torch.zeros(2,1)
b[0,0]=a[0,0]**2+a[1,0]*5
b[1,0]=a[0,0]**3+a[1,0]*4
