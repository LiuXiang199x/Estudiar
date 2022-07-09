from gettext import npgettext
from pkgutil import ImpImporter
import numpy as np
import cv2 as cv
import os
from load_dataset import *
import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image


root_dir = "../source_datasets/PennFudanPed"
root_dir_room = "../source_datasets/slam_maps/room_maps_train"
img_dir  = os.path.join(root_dir, os.listdir(root_dir)[0])
mask_dir  = os.path.join(root_dir, os.listdir(root_dir)[-1])

all_masks = os.listdir(mask_dir)
all_img = os.listdir(img_dir)

img = cv.imread(os.path.join(img_dir, "FudanPed00001.png"))
mask = cv.imread(os.path.join(mask_dir, "FudanPed00001_mask.png"))


def test_visual_mask():
    print(mask.shape)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j][0] == 1 or mask[i][j][1] == 1 or mask[i][j][2] == 1:
                mask[i][j][0] = 255
                mask[i][j][1] = 255
                mask[i][j][2] = 255

            if mask[i][j][0] == 2 or mask[i][j][1] == 2 or mask[i][j][2] == 2:
                mask[i][j][0] = 255
                mask[i][j][1] = 255
                mask[i][j][2] = 0
                
    mm = np.ones(mask.shape)
    cv.imshow("mask", mask)
    cv.imshow("img", img)

    cv.waitKey(0)
    cv.destroyAllWindows()

class Fib():                  #定义类Fib
    def __init__(self,start=0,step=1):
        self.step=step
    def __getitem__(self, key): #定性__getitem__函数，key为类Fib的键
            print("key:::", key)
            a = key+self.step
            print("a:::", a)
            return a          #当按照键取值时，返回的值为a

class LoadDatasets(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
 
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        print("img_path: ", img_path)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path)
        print("img path: ", img_path)
        print("mask path: ", mask_path)
        print(img)
        print(mask)
        print("img shape:::::: ", img.size)
        img = np.array(img)
        print("img shape:::::: ", img.shape)
        print("mask size::::: ", mask.size)
        print(mask.getpixel((0,0)))
        print("===================")

        mask = np.array(mask)
        print(mask[0][0])
        print("mask shape::::: ", mask.shape)
        mask = np.array(mask)
        print("mask shape::::: ", mask.size)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        print(mask.shape)
        print(obj_ids[:, None, None])
        print(obj_ids[:, None, None].shape)
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        print("num_objssss: ", num_objs)
        # print("maskssss: ", masks) 这里报错，masks只有true或者false
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)

dataset = LoadDatasets(root_dir)
dataset_room = LoadDatasets(root_dir_room)


dataset[1]

