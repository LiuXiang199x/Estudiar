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
        f=open('/home/marco/Estudiar/DeepLearning/Project/yolo-v3/data.txt','r')
        self.dataset=f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data=self.dataset[index].strip()
        temp_data=data.split()
        _boxes=np.array([float(x) for x in temp_data[1:]])
        boxes=np.split(_boxes,len(_boxes)//5)

        # 我们选择对图像进行resize，基于左上角去做填充。
        # 比如：640*416，我们会选max生成640*640，然后多余的区域用黑色去填充
        # 虽然改了图像大小，但是坐标其实不用变，因为原点是左上角，而且resize图像也是从左上角开始粘贴的。
        img=make_416_image(os.path.join('/home/marco/Estudiar/DeepLearning/Project/yolo-v3/dataset/img',temp_data[0]))
        
        # 计算缩放比
        w,h=img.size
        case=416/w
        img=img.resize((DATA_WIDTH,DATA_HEIGHT))
        img_data=tf(img)
        labels={}
        for feature_size,_antors in antors.items():
            labels[feature_size]=np.zeros(shape=(feature_size,feature_size,3,5+CLASS_NUM))

            for box in boxes:
                # 得到原图上的坐标
                cls,cx,cy,w,h=box
                # 进行缩放
                cx, cy,w,h=cx*case,cy*case,w*case,h*case

                # 计算偏移量，中心点的偏移量
                _x,x_index=math.modf(cx*feature_size/DATA_WIDTH)
                _y,y_index=math.modf(cx*feature_size/DATA_HEIGHT)
                for i,antor in enumerate(_antors):
                    area=w*h
                    iou=min(area,ANTORS_AREA[feature_size][i])/max(area,ANTORS_AREA[feature_size][i])
                    
                    # 计算 w和h 的偏移量，就是用真实的w和h去除以建议的w和h
                    p_w, p_h = w / antor[0], h / antor[1]
                    labels[feature_size][int(y_index),int(x_index),i]=np.array([iou,_x,_y,np.log(p_w),np.log(p_h),*one_hot(CLASS_NUM,int(cls))])

        return labels[13],labels[26],labels[52],img_data

if __name__ == '__main__':
    dataset=YoloDataSet()
    print(dataset[0][3].shape)  # torch.Size([3, 416, 416])
    print(dataset[0][0].shape)  # (13, 13, 3, 8)
    print(dataset[0][1].shape)  # (26, 26, 3, 8)
    print(dataset[0][2].shape)  # (52, 52, 3, 8)