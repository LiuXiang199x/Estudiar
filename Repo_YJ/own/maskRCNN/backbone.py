import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection import mask_rcnn


model = mask_rcnn.MaskRCNN()
print(model)
