# coding:utf-8

from __future__ import print_function
import torch

x = torch.Tensor(5,3)   # all zeros
# print(x)

x1 = torch.rand(5,3)
# print(x1)
# print(x.size())  # cant us
# e shape(), use size()

# + - * /
# print(x+x1)
# print(torch.add(x,x1))
# print(x.add_(x1))
result = torch.Tensor(5, 3)
torch.add(x, x1, out=result)
# print(result)


