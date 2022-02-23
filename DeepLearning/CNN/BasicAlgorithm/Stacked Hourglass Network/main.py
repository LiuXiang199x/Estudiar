from hourglassNet import hg_module
from layers import *

if __name__ == "__main__":
    hg_mods = nn.ModuleList([
            hg_module(
                5, [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4], 
                 #5表示stack的hourglass网络层数，5个堆叠起来，
                 #[256, 256, 384, 384, 384, 512]表示总网络前半层的维度变化，因为这个网络就是沙漏网络么，越往中间走特征图的维度越高，map的size越小
                make_pool_layer = make_pool_layer,
                make_hg_layer = make_hg_layer
            )])