from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import *
import time
import torch.nn.functional as F

class Tmodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        self.class_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*10*10, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.class_layer(x)
        # print("after conv/pooling: ", x.shape)  # [50, 2], 50是batch size的大小
        
        return x

device =torch.device( "cuda") if torch.cuda.is_available() else torch.device("cpu")
net = Tmodel().to(device)
import torch.optim as optim        
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# print(torch.cuda.device_count())
print(device)
dataTrain, dataTest = load_datas()
for epoch in range(100):
    print("Epoch: ", epoch)
    for iter_num, data in enumerate(dataTrain, 0):
        datas_, labels = data
        datas_ = datas_.to(device)
        labels = labels.to(device)
        # print("datas: ", datas_)
        # print("labels: ", labels)
        
        # 初始化所有参数为0，从0开始train
        optimizer.zero_grad()
        
        outputs = net(datas_)
        # print("outputs shape: ", outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if iter_num%10 == 0:        
            print("loss = ", loss)
    
    if epoch%10 == 0:
        torch.save(net.state_dict(), "/home/agent/Repo_YJ/estudiar/NN_torch/DogCatClassifications/NNN.pth" )