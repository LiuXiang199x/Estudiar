from cProfile import label
from tkinter.tix import LabelEntry
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from MainNet import Net
from torch.autograd  import Variable

import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage

DOWNLOAD = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("===============> Device:",device)

# this model is a simple model with 2 expoch training
# save_model_path = "./AllModels/Lenet_Cifar10.pth"   # 56%
save_model_path = "./AllModels/Lenet_Cifar10_2.pth"  # 59%

show = ToPILImage()         #可以把Tensor转成Image,方便进行可视化
# 把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])
testset = tv.datasets.CIFAR10('./Dataset/CIFAR-10/',train=False,download=DOWNLOAD,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=0)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

net = Net().to(device)
net.load_state_dict(torch.load(save_model_path))
net.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()

print('实际的label: ',' '.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images/2 - 0.5)).resize((400,100))#？？？？？
images = images.to(device)
labels = labels.to(device)
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data,1)#返回最大值和其索引
print('预测结果:',' '.join('%5s'%classes[predicted[j]] for j in range(4)))
correct = 0
total = 0
for data in testloader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total +=labels.size(0)
    correct +=(predicted == labels).sum()
print('10000张测试集中的准确率为: %d %%'%(100*correct/total))
