# coding:utf-8
import torch
from torch.autograd import Variable
# all the neural network is from package "autograd"

x = torch.randn(3)
x = Variable(x, requires_grad = True)   # #需要求导的话，requires_grad=True属性是必须的
# print(x.grad)  # None
# print(x)
y = x * 2
print("y.data.norm() —— L2范数:",y.data.norm())

# 它对张量y每个元素进行平方，然后对它们求和，最后取平方根。 这些操作计算就是L2或欧几里德范数 。
while y.data.norm() < 1000:
	y = y * 2

print("y:", y)  # y: tensor([  906.6727,  -214.8782, -1494.2841], grad_fn=<MulBackward0>)

# FloatTensor 类型转换, 将list ,numpy转化为tensor。此处的y是一维，一行三列
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)   # tensor([ 842.0238,   82.9681, -750.4339], grad_fn=<MulBackward0>)
print(y)  # 在三个不同x下的导数值
print(x.grad)  # tensor([5.1200e+01, 5.1200e+02, 5.1200e-02])


