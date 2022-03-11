
import torch
from PIL import Image
import os
import glob
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Garbage_Loader(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        # (48045, 2), list
        # print(type(self.imgs_info))
        # # <class 'list'>
        # print(len(self.imgs_info)) 
        # # 48045 [[], [], []]
        # print(self.imgs_info[0])
        # # ['/home/agent/垃圾图片库/厨余垃圾_核桃/img_核桃_296.jpeg', '46']
        
        self.train_flag = train_flag
        
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
        
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info
     
    def padding_black(self, img):

        w, h  = img.size

        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

        size_fg = img_fg.size
        size_bg = 224

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img
        
    def __getitem__(self, index):
        print("index:", index)
        print("self.imgs_info[index]:", self.imgs_info[index]) 
        # ['/home/agent/垃圾图片库/厨余垃圾_核桃/img_核桃_296.jpeg', '46']
        
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        # print(len(img))  # 3
        # print("img:", img) # tensor([3, 224, 224])
        # print("label:",label) # 46
        return img, label
 
    def __len__(self):
        print("len(self.imgs_info):", len(self.imgs_info))
        return len(self.imgs_info)
 
    
if __name__ == "__main__":
    train_dataset = Garbage_Loader("/home/agent/Estudiar/DeepLearning/Project/垃圾分类/datasets/train.txt", True)
    print("数据个数：", len(train_dataset))
    print(type(train_dataset)) # <class '__main__.Garbage_Loader'>
    print(len(train_dataset)) # 48045, (tuple: (tensor([3,224,224]), 46), tuple: (tensor([3,224,224]), 46), ...)
    # train_dataset = ([tensor([3,224,224])], 46)
    # train_dataset[0][0]: torch.Size([3, 224, 224])
  
    for item in train_dataset:
        # 这里也可以选择 url， label in train_dataset这种迭代方式，这样item作为一个tuple就被分开了
        # print("item:", item) # tuple: (tensor([3,224,224]), 46)
        print(item)
        print("type(item[0]):", type(item[0]))
        print("item[0].size():", item[0].size()) # torch.Size([3, 224, 224])
        break
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in train_loader:
        print(type(train_loader))
        print("image.shape:", image.shape)  # torch.Size([1, 3, 224, 224])
        print("label:", label)  # label: tensor([93])  # if batch_size = 2: label: tensor([183, 209])
        break
    
    
    