from email.encoders import encode_noop
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2 as cv

class Mydataset(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_infos(txt_path)

        self.train_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),

            ])
        self.val_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
        
        
    def get_infos(self, txt_path):
        txt_file = open(txt_path, "r", encoding="utf-8")
        datas = txt_file.readlines()
        imgs_info = list(map(lambda x:x.strip().split("   "), datas))
        
        return imgs_info
    
    def print_datas(self):
        print(self.imgs_info)
        
    # 编写支持数据集索引的函数，getitem：接收一个index返回图片数据和标签。
    # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
    def __getitem__(self, index):
        url, label = self.imgs_info[index]
        # print("index", index)
        # print(self.imgs_info[index])
        img = Image.open(url).convert("RGB")
        img = self.train_tf(img)
        # label = self.train_tf(label)
        
        return img, label
    # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容


    # 返回数据集大小
    def __len__(self):
        return len(self.imgs_info)


img_path = "/home/agent/Estudiar/DeepLearning/Project/build_load_MyDatas/img_txt/datasets.txt"
c = Mydataset(img_path)
a,b = c.imgs_info[0]
print(a, "___", b)

train_set = DataLoader(dataset=c, batch_size=1)
print(train_set)
for i in train_set:
    print(type(i))
