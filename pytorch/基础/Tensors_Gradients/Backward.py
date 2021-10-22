# coding:utf-8

from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np

a = torch.ones([2, 2], requires_grad=True)
# print(a.is_leaf)   # True
 
b = a + 2
# print(b.is_leaf)    # False
# 因为 b 不是用户创建的，是通过计算生成的
  
x = Variable(torch.Tensor([[2, 3]]), requires_grad=True)   # 叶子节点，用户自己创建生成的
n = Variable(torch.zeros(1,2))   # 此处我们不能requires_grad=True，因为后面需要计算生成。非叶子节点

n[0,0] = x[0,0]**2   # [[2, 3]]
n[0,1] = x[0,1]**3    # [[4, 27]]
# print(x.data)  # [[2, 3]]
# c = np.array([1,2])  # (2,)
# d = np.array([[1,2]])   # (1,2)

# n.backward(x.data)    # 自己手动算，n对x1求偏导=2*X1=4，n对X2求偏导=3*(X2)^2=27
# print(x.grad)  # [[8, 81]]   # 此处算出的结果不太对 8=4*2， 81=27*3
n.backward(torch.FloatTensor([[1,1]]))
print(x.grad)        # [[4, 27]]    这回是对的，那么有理由说明我们传入的参数是导数的系数？也不对！


m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True)
k = Variable(torch.zeros(1, 2))
j = torch.zeros(2 ,2)

k[0, 0] = m[0, 0] ** 2 + 3 * m[0 ,1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
# ( m =(X1=2, X2=3), k=((X1)^2+3*X2, (X2)^2+2*X1)) ———— ((2*X1, 3), (2, 2*X2)) 按理说求完偏导四个结果
# k.backward(torch.FloatTensor([[1, 1]]))
# print(m.grad)  # [[6, 9]]   # 不对，和我们期待的四个结果不一样([4, 3],[2, 6])
# k.backward(parameters)接受的参数parameters必须要和k的大小一模一样，然后作为k的系数传回去
# 正解：k的结构是k=(k1, k2)，传入的参数是(1,1), 1*dk1/dx1 + 1*dk2/dx1 = 1*2*X1 + 1*2 = 6.这样我们就得到了这两个结果，原来我们传入的参数是每次求导的一个系数。
# backward()里面另外的一个参数retain_graph=True，这个参数默认是False，也就是反向传播之后这个计算图的内存会被释放，这样就没办法进行第二次反向传播了，所以我们需要设置为True

# 正解：用k1求偏导
# k.backward(torch.FloatTensor([[1, 0]]))   # 4, 3
# 正解：用k2求偏导
k.backward(torch.FloatTensor([[0, 1]]))   # 2, 6
print(m.grad)




