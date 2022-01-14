import torch
import torch.nn as nn
import torch.nn.functional as F

# 和以前一样, 我们用一个 class 来建立 CNN 模型. 这个 CNN 整体流程是:
# 卷积(Conv2d) -> 激励函数(ReLU) -> 池化, 向下采样 (MaxPooling) -> 再来一遍 -> 展平多维的卷积成的特征图 -> 接入全连接层 (Linear) -> 输出
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
 
    def forward(self, x): 
        # print(x.size())    # 4-1-28-28
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) 
        # print(x.size())    # # 4-6-14-14
        
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # print("x.size():", x.size())   # 4-16-5-5
        
        x = x.view(x.size()[0], -1)   #展开成一维的
        # print("x.size():", x.size())   # 4-400
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x