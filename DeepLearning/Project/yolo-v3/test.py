from random import triangular
import torch
from torch import mode
from yolo_v3_net import *
from dataset import *
from torch.utils.data import DataLoader

model_path = "/home/agent/Estudiar/DeepLearning/Project/yolo-v3/params/net.pt"

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

a = torch.randn(2,3,5)
b = a.permute(1,2,0)
print(a)
print(b)
print(b.shape)