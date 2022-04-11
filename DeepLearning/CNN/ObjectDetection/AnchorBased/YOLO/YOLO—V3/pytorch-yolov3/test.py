import torch

x=torch.randn(1,13,13,3,8)

mask=x[...,0]>1
index=mask.nonzero()
print(mask)
print(mask.shape)
print(index)
print(index.shape)
print(x[mask].shape)