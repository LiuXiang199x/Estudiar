import imp
from xmlrpc.client import TRANSPORT_ERROR
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


root_dir = "/home/marco/Estudiar/Repo_YJ/classificationDogsCats/datas"
eval_dir = "/home/marco/Estudiar/Repo_YJ/classificationDogsCats/datas/eval"

dogs_dir = os.path.join(root_dir, "dogs")
cats_dir = os.path.join(root_dir, "cats")

LABELS_ = {"dogs": 0, "cats": 1}

def initDatas(dataPath, label):
    data = []
    lstdir = os.listdir(dataPath)
    for name in lstdir:
        data.append([os.path.join(dataPath, name), label])
    return data

transforms_ = transforms.Compose([
    # transforms.CenterCrop(224)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

class AnimalDataset(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index: int):
        img, label = self.data[index]
        img = Image.open(img).convert("RGB")
        img = self.transform(img)

        return img, label
    
    def __len__(self) -> int:
        return len(self.data)

def loadDatasTrain():

    dogsTest = initDatas(dogs_dir, 0)
    catsTest = initDatas(cats_dir, 1)

    dogsTest = AnimalDataset(dogsTest, transforms_)
    catsTest = AnimalDataset(catsTest, transforms_)
    
    allTest = dogsTest+catsTest
    testLoader = DataLoader(allTest, batch_size=12, shuffle=True, num_workers=0, pin_memory=True)
    
    return testLoader

def loadDatasEval():

    dogsTest = initDatas(os.path.join(eval_dir, "dogs"), 0)
    catsTest = initDatas(os.path.join(eval_dir, "cats"), 1)

    dogsTest = AnimalDataset(dogsTest, transforms_)
    catsTest = AnimalDataset(catsTest, transforms_)
    
    allTest = dogsTest+catsTest
    testLoader = DataLoader(allTest, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    
    return testLoader