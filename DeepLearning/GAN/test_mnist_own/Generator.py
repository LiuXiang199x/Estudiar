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

####### 定义生成器 Generator #####
#输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(100,256),#用线性变换将输入映射到256维
            nn.ReLU(True),#relu激活
            nn.Linear(256,256),#线性变换
            nn.ReLU(True),#relu激活
            nn.Linear(256,784),#线性变换
            nn.Tanh()#Tanh激活使得生成数据分布在【-1,1】之间
        )
    def forward(self, x):
        x=self.gen(x)
        return x