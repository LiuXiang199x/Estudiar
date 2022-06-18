from matplotlib.pyplot import axis
import numpy as np
import cv2 as cv
import torch
from einops import rearrange, repeat
#

a = torch.randn((2,3))
print(a)
b = torch.zeros(1,3)
print(b)
c = torch.ones(2,1)
print(c)
print(torch.cat([a, b], axis=0))
print(torch.cat([a, c], axis=1))