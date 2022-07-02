import numpy as np
import torch
import torch.nn as nn


def concatenate():
    a1 = np.array([[1,2,3]])
    a2 = np.array([[3,3,3]])
    print(np.concatenate((a1, a2), axis=0))
    print(np.concatenate((a1, a2), axis=1))


    