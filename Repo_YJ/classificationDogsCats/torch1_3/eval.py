from operator import mod
from unittest import TestLoader
import torch
import torch.nn as nn
import tornado
from datasets_laptop import loadDatasEval, loadDatasTrain
from Network import Tmodel
import torch.nn as nn

pth_ = "/home/marco/Estudiar/Repo_YJ/classificationDogsCats/torch1_3/model_14000.pth"

# testLoader = loadDatasEval()
testLoader = loadDatasTrain()
model = Tmodel()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(torch.load(pth_).keys())
print(torch.load(pth_)["epoch"])
model.load_state_dict(torch.load(pth_)["state_dict"])


softmax = nn.Softmax()
total = 0
counter = 0
for item in testLoader:
    total += 1
    print("====================")
    data, label = item
    data = data.to(device)
    print("label: ", label)
    label = label.to(device)
    output = softmax(model(data)[0])
    print(output)
    output = torch.max(output, 0)[1]
    if output==label:
        counter += 1
    
print("precision: ", counter/total)