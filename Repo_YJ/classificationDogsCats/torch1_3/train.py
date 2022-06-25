import torch
import torch.nn as nn
from datasets_laptop import loadDatasTrain
from Network import Tmodel


EPOCHES = 100

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tmodel().to(device)
    datas = loadDatasTrain()
    
    criterion = nn.CrossEntropyLoss()
    optm = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



train()