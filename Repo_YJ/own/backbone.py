import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection import mask_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def getNetwork(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, 
    num_classes=num_classes, pretrained_backbone=False)


    return model

net = getNetwork(2)
print(net)