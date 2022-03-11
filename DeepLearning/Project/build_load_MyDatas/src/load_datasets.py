from email.encoders import encode_noop
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
        
        
        