from nbformat import write
import torch
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter

pp = models.resnet50()
fake_img = torch.rand(1, 3, 200, 200)
print(pp(fake_img))

writer = SummaryWriter()
writer.add_graph(pp, fake_img)