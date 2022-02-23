import torch
import torch.nn as nn
from residual import Residual

class HourGlass(nn.Module):
    """不改变特征图的高宽"""
    def __init__(self,n=4,f=128):
        """
        :param n: hourglass模块的层级数目
        :param f: hourglass模块中的特征图数量
        :return:
        """
        super(HourGlass,self).__init__()
        self._n = n
        self._f = f
        self._init_layers(self._n,self._f)

    def _init_layers(self,n,f):
        # 上分支
        setattr(self,'res'+str(n)+'_1',Residual(f,f))
        # 下分支
        setattr(self,'pool'+str(n)+'_1',nn.MaxPool2d(2,2))
        setattr(self,'res'+str(n)+'_2',Residual(f,f))
        if n > 1:
            self._init_layers(n-1,f)
        else:
            self.res_center = Residual(f,f)
        setattr(self,'res'+str(n)+'_3',Residual(f,f))
        setattr(self,'unsample'+str(n),Upsample(scale_factor=2))


    def _forward(self,x,n,f):
        # 上分支
        up1 = x
        up1 = eval('self.res'+str(n)+'_1')(up1)
        # 下分支
        low1 = eval('self.pool'+str(n)+'_1')(x)
        low1 = eval('self.res'+str(n)+'_2')(low1)
        if n > 1:
            low2 = self._forward(low1,n-1,f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.'+'res'+str(n)+'_3')(low3)
        up2 = eval('self.'+'unsample'+str(n)).forward(low3)

        return up1+up2

    def forward(self,x):
        return self._forward(x,self._n,self._f)
