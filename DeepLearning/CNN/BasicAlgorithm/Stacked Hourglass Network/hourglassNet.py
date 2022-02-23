import torch
import torch.nn as nn
from layers import *
from layers import _make_layer, _make_pool_layer, _make_merge_layer, _make_layer_revr, _make_unpool_layer
from residual import residual

class hg_module(nn.Module):
    def __init__(self, n, dims, modules,      
        make_up_layer = _make_layer,
        make_pool_layer = _make_pool_layer, 
        make_hg_layer = _make_layer, 
        make_low_layer = _make_layer, 
        make_hg_layer_revr = _make_layer_revr,
        make_unpool_layer = _make_unpool_layer, 
        make_merge_layer = _make_merge_layer):   
        #这块就是一些ModuleList，换了个名字，网络结构流看forward函数很清晰
        #moudle=[2, 2, 2, 2, 2, 4]
        #dim=[256, 256, 384, 384, 384, 512]
        
        super(hg_module, self).__init__()

        curr_mod = modules[0]  # moudule列表的每个数字 表示MoudleList要重复用几次
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n    = n
        self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod)  #用1次，看第2步make_up_layer函数
        #up1 的意思是接收外层的传入的特征图，若外层是1层，内层就是第二层（这里刚开始是输入图像----->第一层），up1层做residual操作特征图尺寸没变
        self.max1 = make_pool_layer(curr_dim)  # 注意啦，这里尺寸降了一次
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)  # resiual 操作，特征map尺寸不变，变的是特征map的维度个数
        self.low2 = hg_module(
            n - 1, dims[1:], modules[1:],  #递归啦，这里就是由第2层往第3层走， 3-->4，  4--->5
                                           #map的维度由内层256--->256, 256---->384, 384----->384, 384------>384,  384------>512
            make_up_layer=make_up_layer,      
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)  #n=1时，跳出来直接到了最外层了
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2  = make_unpool_layer(curr_dim) # 最外层upsample一次，因为递归开始前，输入图像------->第一层 降维了一次，
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        merg = self.merg(up1, up2)
        return merg
