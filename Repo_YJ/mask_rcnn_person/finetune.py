from operator import mod
from xmlrpc.client import TRANSPORT_ERROR
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

print(model)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

print(model)