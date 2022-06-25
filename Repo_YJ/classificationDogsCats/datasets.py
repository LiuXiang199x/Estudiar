from cgi import test
from xmlrpc.client import TRANSPORT_ERROR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

labels_ = {"Dog": 0, "Cat": 1}
print("pytorch 的数据加载到模型的操作顺序是这样的：\n\
            ① 创建一个 Dataset 对象(存路径和label，用__getitem__来迭代)\n\
            ② 创建一个 DataLoader 对象(将数据集对象打包成torch接受的方式tensor/gpu等等)\n\
            ③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练\n\
            dataset = MyDataset()\n\
            dataloader = DataLoader(dataset)\n\
            num_epoches = 100\n\
            for epoch in range(num_epoches):\n\
                for img, label in dataloader:\n\
                    ....")

print("获取Datasets流程如下: \n\
    1. 将数据路径和label一起打包好为list，如：[[../../../XX.jpg, lable], [../../../XX.jpg, lable]...]  \n\
    2. 对数据进行划分，多少用来train，多少用来test。 \n\
    3. 使用transform对数据进行resize，ToTensor，rotation等操作。 \n\
    4. 使用Datasets中的__getitem__()来真正的读取数据创建数据集，再通过DataLoader打包成batch准备输入网络。")

##################################################
# 12501张图片，0-9999 ——> train；10000-12501 ——> test
##################################################
root_dir = "/home/agent/桌面/data/PetImages"
dirs = os.listdir(root_dir)
dogs_dir = os.path.join(root_dir, "Dog")
cats_dir = os.path.join(root_dir, "Cat")
total_dogs = len(os.listdir(dogs_dir)) - 1
total_cats = len(os.listdir(cats_dir)) - 1
print("The total nums of dogs: ", total_dogs)
print("The total nums of cats: ", total_cats)

##################################################################
# 数据预处理：得到一个包含所有图片文件名（包含路径）和标签（狗1猫0）的列表：
# [["../../../XX.jpg", 1], ["../../../XX.jpg", 1], ["../../../XX.jpg", 1], ["../../../XX.jpg", 1]]
##################################################################
def init_imgs(path, lens, label_name: str):
    data = []
    name = labels_[label_name]
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])
        
    return data

dataDog_train = init_imgs(os.path.join(dogs_dir, "%d.jpg"), [0, 10000], "Dog")
dataDog_test = init_imgs(os.path.join(dogs_dir, "%d.jpg"), [10000, 12500], "Dog")
dataCat_train = init_imgs(os.path.join(cats_dir, "%d.jpg"), [0, 10000], "Cat")
dataCat_test = init_imgs(os.path.join(cats_dir, "%d.jpg"), [10000, 12500], "Cat")

data2train = dataDog_train + dataCat_train
data2test = dataDog_test + dataCat_test
# print(dataDog_train)
# print(dataDog_test)
# print(dataCat_train)
# print(dataCat_test)

#################################################################
# transform:
# 1）、transforms.CenterCrop(224)，从图像中心开始裁剪图像，224为裁剪大小
# 2）、transforms.Resize((224, 224)),重新定义图像大小
# 3）、 transforms.ToTensor()，很重要的一步，将图像数据转为Tensor
# 4）、transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))，归一化
#################################################################
init_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# 重写pytorch的Dataset类
# __getitem__是真正读取数据的地方，迭代器通过索引来读取数据集中数据，因此只需要这一个方法中加入读取数据的相关功能即可。
# 在这个函数里面，我们对第二步处理得到的列表进行索引，接着利用第三步定义的Myloader来对每一个路径进行处理，
# 最后利用pytorch的transforms对RGB数据进行处理，将其变成Tensor数据。

class MyDataset(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()
        self.data = data
        self.transform = transform
    
    def __getitem__(self,item):
        img, label = self.data[item]
        img = Image.open(img).convert("RGB")
        img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.data)

    
def load_datas():
    dataDog_train = init_imgs(os.path.join(dogs_dir, "%d.jpg"), [0, 10000], "Dog")
    dataDog_test = init_imgs(os.path.join(dogs_dir, "%d.jpg"), [10000, 12500], "Dog")
    dataCat_train = init_imgs(os.path.join(cats_dir, "%d.jpg"), [0, 10000], "Cat")
    dataCat_test = init_imgs(os.path.join(cats_dir, "%d.jpg"), [10000, 12500], "Cat")

    data2train = dataDog_train + dataCat_train
    data2test = dataDog_test + dataCat_test
    data2train = MyDataset(data2train, init_transform)
    data2test = MyDataset(data2test, init_transform)

    trainLoader = DataLoader(dataset=data2train, batch_size=50, shuffle=True, num_workers=0, pin_memory=True)
    testLoader = DataLoader(dataset=data2test, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    return trainLoader, testLoader

# # 试一试类  和  dataloader区别
# print("在继承的class\"Dataset\"中，把图像按照[路径, label]读入后转为[Image.RGB, label], 经过class中的\n\
#     transform把图像数据转为tensor和进行一些归一化。但是此过程没有把label做一些改变。\n\
#     相反DataLoader中把class中数据掉进了重新封装为torch可以用的接口，把label打包为tensor，加上gpu，shuffle,batchSize等操作.")
# a, _, b, d = load_datas()
# # print(a)   # <__main__.MyDataset object at 0x7ff575f12250>
# # print(len(a))  # 20000
# # print(type(a))    # <class '__main__.MyDataset'>
# # print("=========")
# # print(b)    # <torch.utils.data.dataloader.DataLoader object at 0x7ff650b0bd30>
# # print(len(b))  # 20000 
# # print(type(b))    # <class 'torch.utils.data.dataloader.DataLoader'>
# print(d)    # <torch.utils.data.dataloader.DataLoader object at 0x7ff650b0bd30>
# print(len(d))  # 50   因为testLoader的 batchsize = 100；总共test数据是5000个，所以长度是50
# # d的形状：第一个有四层括号，从里到外分别是：[[[[通道]*3]_图片*N]_BatchSize]
# # [ [ [[..], [..], [..]], [[..], [..], [..]], ...], [ [[..], [..], [..]], [[..], [..], [..]], ...] ......] , tensor(lable)[0, 1, 0, ...]
# print(type(d))    # <class 'torch.utils.data.dataloader.DataLoader'>

# print("-------------------------------------------------------")
