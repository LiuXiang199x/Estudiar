import numpy as np
import torch
import torch.nn as nn


def concatenate():
    a1 = np.array([[1,2,3]])
    a2 = np.array([[3,3,3]])
    print(np.concatenate((a1, a2), axis=0))
    print(np.concatenate((a1, a2), axis=1))

    a3 = np.array([[2,3,4],[1,2,3]])
    print(a3)
    print(a3.T)
    
concatenate()