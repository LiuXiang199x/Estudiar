import torch
import torch.nn as nn
from torchvision import models

models.detection.fasterrcnn_resnet50_fpn

model = models.detection.maskrcnn_resnet50_fpn()
# model ======> 1. backbone(5 layers) 
                        # 2. FPN: inner_blocks : layer_blocks : extra_blocks
                        # 3. RPN(): anchor_generator : head(RPNHead - cls_logits - bbox_pred)
                        # 4. roi heads:
                            # 4.1 box_roi_pool
                            # 4.2 box_head
                            # 4.3 box_predictor(cls_score, bbox_pred)
                            # 4.4 mask_roi_pool
                            # 4.5 mask_head
                            # 4.6 maks_predictor
print(model)