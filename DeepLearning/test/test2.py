import torch
import torch.nn as nn
from torchvision.models import alexnet
from tensorboardX import SummaryWriter

fake_img = torch.rand(1, 3, 230, 230)
a = alexnet()
print(a(fake_img))

writer = SummaryWriter(comment='test_your_comment',filename_suffix="_test_your_filename_suffix")
writer.add_graph(a, fake_img)  # 模型及模型输入数据