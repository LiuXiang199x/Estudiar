import torch
import torch.nn as nn
from datasets_laptop import loadDatasEval
from Network import Tmodel

pth_ = "/home/marco/Estudiar/Repo_YJ/classificationDogsCats/torch1_3/model_14000.pth"

testLoader = loadDatasEval()
model = Tmodel()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(torch.load(pth_).keys())
print(torch.load(pth_)["epoch"])
model.load_state_dict(torch.load(pth_)["state_dict"])

for item in testLoader:
    data, label = item
    data = data.to(device)
    label = label.to(device)
    print(model(data))
    break
