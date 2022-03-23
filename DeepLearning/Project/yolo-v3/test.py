from random import triangular
from re import L
import torch
from torch import mode
from yolo_v3_net import *
from dataset import *
from torch.utils.data import DataLoader
from config import *

# model_path = "/home/agent/Estudiar/DeepLearning/Project/yolo-v3/params/net.pt"

# params = torch.load(model_path)
# print(type(params))
# print(params["detetion_52.1.bias"])

# model = Yolo_V3_Net()
# model.load_state_dict(params)

# print("loadding done")

# model.eval()


# test_dataset = YoloDataSet()
# test_dataset = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)
# for i in test_dataset:
#     print(type(i))  # list  # (4, 2, 13, 13)
#     break

for i in antors:
    print(antors)
    
for a,b in antors.items():
    print(a)
    print(b)
