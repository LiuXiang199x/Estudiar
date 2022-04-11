import math
import os

import torch

from config import *
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import *
from torchvision import transforms

tf=transforms.Compose([
    transforms.ToTensor()
])

def one_hot(cls_num,i):
    rst=np.zeros(cls_num)
    rst[i]=1
    return rst

class YoloDataSet(Dataset):
    def __init__(self):
        f=open('data.txt','r')
        self.dataset=f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data=self.dataset[index].strip()
        temp_data=data.split()
        _boxes=np.array([float(x) for x in temp_data[1:]])
        boxes=np.split(_boxes,len(_boxes)//5)
        img=make_416_image(os.path.join('data\\images',temp_data[0]))
        w,h=img.size
        case=416/w
        img=img.resize((DATA_WIDTH,DATA_HEIGHT))
        img_data=tf(img)
        labels={}
        for feature_size,_antors in antors.items():
            labels[feature_size]=np.zeros(shape=(feature_size,feature_size,3,5+CLASS_NUM))

            for box in boxes:
                cls,cx,cy,w,h=box
                cx, cy,w,h=cx*case,cy*case,w*case,h*case
                _x,x_index=math.modf(cx*feature_size/DATA_WIDTH)
                _y,y_index=math.modf(cx*feature_size/DATA_HEIGHT)
                for i,antor in enumerate(_antors):
                    area=w*h
                    iou=min(area,ANTORS_AREA[feature_size][i])/max(area,ANTORS_AREA[feature_size][i])
                    p_w, p_h = w / antor[0], h / antor[1]
                    labels[feature_size][int(y_index),int(x_index),i]=np.array([iou,_x,_y,np.log(p_w),np.log(p_h),*one_hot(CLASS_NUM,int(cls))])

        return labels[13],labels[26],labels[52],img_data

if __name__ == '__main__':
    dataset=YoloDataSet()
    print(dataset[0][3].shape)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
