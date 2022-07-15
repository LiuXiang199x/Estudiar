import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import faster_rcnn
from torch.utils.data import DataLoader

model = faster_rcnn.FasterRCNN()  # 必须要指定一个backbone
print(model)
# #