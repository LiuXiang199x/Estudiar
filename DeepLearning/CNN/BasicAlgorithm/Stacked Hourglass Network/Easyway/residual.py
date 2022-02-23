import torch
import torch.nn as nn

class Residual(nn.Module):
    """
    残差模块，并不改变特征图的宽高
    """
    def __init__(self,ins,outs):
        super(Residual,self).__init__()
        # 卷积模块
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins,outs/2,1),
            nn.BatchNorm2d(outs/2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs/2,outs/2,3,1,1),
            nn.BatchNorm2d(outs/2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs/2,outs,1)
        )
        # 跳层
        self.skipConv = nn.Conv2d(ins,outs,1)

    def forward(self,x):
        residual = x
        x = self.convBlock(x)
        residual = self.skipConv(residual)
        x += nn.ReLU(residual)
        return x
