from modulefinder import IMPORT_NAME
import torch
import torch.nn as nn
from VGG import VGG
from YOLO_V1 import YOLOV1

if __name__ == '__main__':
    print("========= TEST IN VGG =========")
    vgg = VGG()
    x  = torch.randn(1,3,512,512)
    feature,x_fea,x_avg = vgg(x)
    print("Just in VGG___After VGG before yolo -> feature:", feature.shape)
    print("Just in VGG___After VGG before yolo -> x_fea:", x_fea.shape)
    print("Just in VGG___After VGG before yolo -> x_avg:", x_avg.shape)
 
    print("========= TEST IN YOLO-V1 =========")
    yolov1 = YOLOV1()
    feature = yolov1(x)
    # feature_size b*7*7*30
    print("After yolo -> feature:", feature.shape)
    print("可以看到yolo-v1和vgg16的网络结构几乎都是一样的！前13个都是同样卷基层，之后后面分类的时候（全连接层）不一样")