import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import faster_rcnn

b = torchvision.models.resnet18()
c = torchvision.models.resnet50()
d = torchvision.models.mobilenet_v2()
e = torchvision.models.vgg16_bn()

# model = faster_rcnn.FasterRCNN()  # 必须要指定一个backbone
print(e.children())
