# coding:utf-8

import torch

a = torch.Tensor(5)   # all zeros: [4.7429e+30, 7.1354e+31, 7.1118e-04, 1.7444e+28, 7.3909e+22])
a0 = torch.zeros(5)  # all zeros:[0, 0, ...]
a1 = torch.ones(5)
# print(a0)

# tensor 2 array
b = a1.numpy()

a1.add_(1)  # [2., 2., 2., 2., 2.]
# print(a1) # [2., 2., 2., 2., 2.]
# print(b) # [2., 2., 2., 2., 2.]

# array 2 tensors
import numpy as np
aa = np.ones(5)
bb = torch.from_numpy(aa)
np.add(aa, 1, out=aa)
# print(aa)
# print(bb)   # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

