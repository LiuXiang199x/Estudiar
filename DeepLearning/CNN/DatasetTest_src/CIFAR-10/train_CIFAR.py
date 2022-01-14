import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from MainNet import Net
from tensorboardX import SummaryWriter

import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage

DOWNLOAD = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("===============> Device:",device)
save_model_path = "./AllModels/Lenet_Cifar10_2.pth"
writer = SummaryWriter(logdir="tensorboard", comment="CIFAR-10-2")


# 使用torchvision加载并预处理CIFAR10数据集
show = ToPILImage()         #可以把Tensor转成Image,方便进行可视化
# 把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])
trainset = tv.datasets.CIFAR10(root='./Dataset/CIFAR-10/',train = True,download=DOWNLOAD,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
(data,label) = trainset[100]
print(classes[label])#输出ship
show((data+1)/2).resize((100,100))
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400,100))#make_grid的作用是将若干幅图像拼成一幅图像

net = Net().to(device)
print(net)

#定义损失函数和优化器
from torch import optim
criterion  = nn.CrossEntropyLoss()#定义交叉熵损失函数
optimizer = optim.SGD(net.parameters(),lr = 0.0001,momentum=0.9)

#训练网络
from torch.autograd  import Variable
for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):#enumerate将其组成一个索引序列，利用它可以同时获得索引和值,enumerate还可以接收第二个参数，用于指定索引起始值
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss  = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        writer.add_scalar("train_loss", loss.item(), epoch)

        running_loss += loss.item()
        if i % 2000 ==1999:
            print('[%d, %5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0

torch.save(net.state_dict(), save_model_path)