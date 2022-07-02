import numpy as np
import torch
import torch.nn as nn


# 最全能，根据指定的维度，对一个元组、列表中的list或者ndarray进行连接
def concatenate():
    a1 = np.array([[1,2,3]])
    a2 = np.array([[3,3,3]])
    print(np.concatenate((a1, a2), axis=0))
    print(np.concatenate((a1, a2), axis=1))

    a3 = np.array([[2,3,4],[1,2,3]])
    print(a3)
    print(a3.T)

# 将一堆数组的数据按照指定的维度进行堆叠。
def stack():
    a1 = np.array([[1,2,3]])
    a2 = np.array([[3,3,3]])

    print(np.stack((a1,a2), axis=1))
    
    print("\n======== vstack ========")
    # 它是垂直（按照行顺序）的把数组给堆叠起来。
    print(np.vstack((a1, a2)))

    print("\n======== hstack ========")
    # 它其实就是水平(按列顺序)把数组给堆叠起来
    print(np.hstack((a1, a2)))


concatenate()
stack()