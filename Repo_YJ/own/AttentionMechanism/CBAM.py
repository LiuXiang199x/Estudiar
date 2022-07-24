import torch
import torch.nn as nn
from CA import *

class CBAM(nn.Module):
    def __init__(self, inplanes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplanes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x*self.ca(x)
        result = out*self.sa(out)
        return result