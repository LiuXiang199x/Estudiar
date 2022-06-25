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



testLoader = loadDatas()
model = Tmodel()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("/home/marco/Estudiar/Repo_YJ/classificationDogsCats/NNN.pth"))

