# coding:utf-8

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# nn建立在autograd的基础上来进行模型的定义和微分

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:] # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = Net()
print(net)

'''
神经网络的输出结果是这样的
Net (
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear (400 -> 120)
  (fc2): Linear (120 -> 84)
  (fc3): Linear (84 -> 10)
)
'''
