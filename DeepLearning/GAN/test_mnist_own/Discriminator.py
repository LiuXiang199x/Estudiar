import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

#定义判别器  #####Discriminator######使用多层网络来作为判别器
#将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(784,256),#输入特征数为784，输出为256
            nn.LeakyReLU(0.2),#进行非线性映射
            nn.Linear(256,256),#进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()#也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )
    def forward(self, x):
        x=self.dis(x)
        return x
    