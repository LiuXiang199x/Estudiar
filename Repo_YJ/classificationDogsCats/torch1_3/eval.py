import imp
from xmlrpc.client import TRANSPORT_ERROR
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from datasets_laptop import *


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

testLoader = loadDatas()
model = Tmodel()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("/home/marco/Estudiar/Repo_YJ/classificationDogsCats/NNN.pth"))

