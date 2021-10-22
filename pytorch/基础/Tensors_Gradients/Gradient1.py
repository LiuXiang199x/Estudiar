# coding:utf-8
import torch
from torch.autograd import Variable
# all the neural network is from package "autograd"

"""
autograd 包提供Tensor所有操作的自动求导方法。这是一个运行时定义的框架，这意味着你的反向传播是根据你代码运行的方式来定义的，因此每一轮迭代都可以各不相同。
pytorch中 变量variable有三个属性attribute：data，grad，creator（creator——grad_fn）
data：我们被包裹起来的数据；creator：用来记录通过何种计算得到当前的variable；grad：在进行反向传播的时候用来记录数据的梯度的。
creator ——（每个变量都有一个.creator属性，此属性会告诉我们“得到此变量所进行的操作”。如果这个变量是我们自己直接创建的，则这个变量的creator属性为None。）
autograd.Variable 这是这个包中最核心的类。 它包装了一个Tensor，并且几乎支持所有的定义在其上的操作。一旦完成了你的运算，你可以调用 .backward()来自动计算出所有的梯度。你可以通过属性 .data 来访问原始的tensor，而关于这一Variable的梯度则集中于 .grad 属性中。autograd.Variable —— (data; grad; creator)
还有一个在自动求导中非常重要的类 Function。Variable 和 Function 二者相互联系并且构建了一个描述整个运算过程的无环图。每个Variable拥有一个 .creator 属性，其引用了一个创建Variable的 Function。(除了用户创建的Variable其 creator 部分是 None)。
如果你想要进行求导计算，你可以在Variable上调用.backward()。 如果Variable是一个标量（例如它包含一个单元素数据），你无需对backward()指定任何参数，然而如果它有更多的元素，你需要指定一个和tensor的形状想匹配的grad_output参数。
"""

x = Variable(torch.ones(2,2), requires_grad=True)
y = x + 2
# print('x: ', x)
# print('y: ', y)   # [[3, 3], [3,3]]
# print(x.grad_fn)  # None
# print(y.grad_fn)  # <AddBackward0 object at 0x7fcc4ef4aed0>

# y 是作为一个操作的结果创建的因此y有一个creator
z = y * y * 3    # y*y= tensor([[9., 9.],[9., 9.]], grad_fn=<MulBackward0>)

out = z.mean()   # tensor(27., grad_fn=<MeanBackward0>)  Zi = 3*(Xi+2)**2 | Xi=1 = 27，导数是3(Xi+2)/2  (因为这里求平均除以的4)

# 现在我们来使用反向传播, 此处的 out是一个数字, 标量
out.backward()
# out.backward(torch.Tensor([1.0]))
# out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
# 在此处输出 d(out)/dx
print(x.grad)  # tensor([[4.5000, 4.5000], [4.5000, 4.5000]])


a = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
b = a + 3
c = b * b * 3
out = c.mean()
out.backward()

print('=====simple gradient======')
print('input')
print(a.data)
print('compute result is')
print(out.data)   # tensor(91.5000)
print(out)  # tensor(91.5000, grad_fn=<MeanBackward0>)
print('input gradients are')
print(a.grad)



