from pyexpat import model
import importlib_metadata
import torch
import torchvision.models as models

for name in models.__dict__:
    print(name)
    print(callable(models.__dict__[name]))
    
