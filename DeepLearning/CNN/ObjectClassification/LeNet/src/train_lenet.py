from numpy.lib.arraysetops import union1d
import torch
import torch.nn as nn
import torchvision
import os
from LeNet_Net import CNN
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter


# 然后定义一个SummaryWriter() 实例。看到SummaryWriter()的参数为：def __init__(self, log_dir=None, comment='', **kwargs):
# log_dir = os.path.join("tensorboard", "Loss")
loss_writer = SummaryWriter(log_dir="tensorboard", comment="Loss")
# log_dir为生成的文件所放的目录，comment为文件名称。

save_model_path = "/home/agent/Estudiar/pytorch/深度学习/CNN/图像识别/LeNet/models/Lenet5_gpu.pth"

# Free condition:  GPU 374/440, CPU:0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # 324s CPU 103%, GPU 880/4400
# device = torch.device("cpu")   # 767s     # cpu 750%     gpu:350/4400
print(device)

net = CNN().to(device)
print(net)

EPOCH = 1
batch_size = 4
LR = 0.001
DOWNLOAD_MINIST = False


transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)), # 归一化
                             ])
 
train_dataset = torchvision.datasets.MNIST(root='../mnist', 
                            train=True, 
                            transform=transform, 
                            download=DOWNLOAD_MINIST)
test_dataset = torchvision.datasets.MNIST(root='../mnist', 
                            train=False, 
                            transform=transform,
                            download=DOWNLOAD_MINIST)
 

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


print(train_dataset)
print(test_dataset)

# plot one example: training data的第一张图片
"""
print(train_data.train_data.size())   # torch.Size([60000, 28, 28])
print(train_data.train_labels.size())  # torch.Size([60000])
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()
"""


from torch import optim
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 优化器
 
import time
 
start_time = time.time()

for epoch in range(100):
    running_loss = 0.0 #初始化loss
    for i, (inputs, labels) in enumerate(trainloader, 0):
 
        # 输入数据
        inputs =  inputs.to(device)
        labels =  labels.to(device)
 
        # 梯度清零
        optimizer.zero_grad()
 
        # forward + backward 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()   
 
        # 更新参数 
        optimizer.step()
 
        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 2000 == 1999: # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0

        # 第一个参数：保存图片的名称； 第二个参数：Y轴数据； 第三个参数：X轴数据
        loss_writer.add_scalar("Loss", loss.item(), epoch)

stop_time = time.time()
print('Finished Training 耗时： ', (stop_time - start_time), '秒')
torch.save(net.state_dict(), save_model_path)