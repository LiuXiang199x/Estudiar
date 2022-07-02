import torch
import torch.nn as nn
import torchvision.models.detection.rpn as rpn


# RegionProposalNetwork是整个rpn的主体，其中集成了AnchorGenerator和RPNHead，
# 功能包含生成anchors，anchor与groundtruth的匹配，nms，回归与分类损失的计算等等。