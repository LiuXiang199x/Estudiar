from numpy.lib.arraysetops import union1d
import torch
import torch.nn as nn
import torchvision
from LeNet_Net import CNN
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
from torchvision import datasets, transforms

save_model_path = "../models/Lenet5_cpu.pth"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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

for epoch in range(10):
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

stop_time = time.time()
print('Finished Training 耗时： ', (stop_time - start_time), '秒')
torch.save(net.state_dict(), save_model_path)