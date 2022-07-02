from typing import OrderedDict
import torch
import torch.nn as nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

def fpnAPI():
    m = FeaturePyramidNetwork([10, 20, 30], 5)
    # get some dummy data
    x = OrderedDict()
    x['feat0'] = torch.rand(1, 10, 64, 64)
    x['feat2'] = torch.rand(1, 20, 16, 16)
    x['feat3'] = torch.rand(1, 30, 8, 8)
    print(x)
    # compute the FPN on top of x
    output = m(x)
    print(output.shape)
    # print([(k, v.shape) for k, v in output.items()])
    
    
    