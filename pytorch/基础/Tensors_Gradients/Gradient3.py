# coding:utf-8
import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([1, 2, 3]), requires_grad=True)
d = torch.mean(x)
print(d)

d.backward()    # 一个y对应三个x
print(x.grad)  # tensor([0.3333, 0.3333, 0.3333])
# 叠加的
d.backward()
print(x.grad)   # tensor([0.6667, 0.6667, 0.6667])

# 三个y对应三个x
c = x*2
print(c)  # tensor([2., 4., 6.], grad_fn=<MulBackward0>)
c.backward(torch.Tensor([1, 0.01, 0.001]))
print(x.grad)  # tensor([2.6667, 0.6867, 0.6687])

# 总结：backward参数是否必须取决于因变量的个数，从数据中表现为标量和矢量；如果输出是一个标量的话，backward不需要任何输入。k.backward(parameters)接受的参数parameters必须要和k的大小一模一样，然后作为k的系数传回去。我们传入的参数是每次求导的一个系数。
