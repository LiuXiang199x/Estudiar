# 数据预处理
# 在计算机视觉中，通常会对图像做一些随机的变化，产生相似但又不完全相同的样本。
# 主要作用是扩大训练数据集，抑制过拟合，提升模型的泛化能力，常用的方法主要有以下几种：
    # 随机改变亮暗、对比度和颜色
    # 随机填充
    # 随机裁剪
    # 随机缩放
    # 随机翻转
    # 随机打乱真实框排列顺序
# 下面我们分别使用numpy 实现这些数据增强方法。

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
from ReadPreprocessImg import get_insect_names,get_annotations,get_bbox,get_img_data_from_file

TRAINDIR = '/home/marco/Datasets/Paddle/insects/train'
TESTDIR = '/home/marco/Datasets/Paddle/insects/test'
VALIDDIR = '/home/marco/Datasets/Paddle/insects/val'

# 随机改变亮暗、对比度和颜色等
def random_distort(img):
    # 随机改变亮度
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)
    # 随机改变对比度
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)
    # 随机改变颜色
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img

# 定义可视化函数，用于对比原图和图像增强的效果
import matplotlib.pyplot as plt
def visualize(srcimg, img_enhance):
    # 图像可视化
    plt.figure(num=2, figsize=(6,12))
    plt.subplot(1,2,1)
    plt.title('Src Image', color='#0000FF')
    plt.axis('off') # 不显示坐标轴
    plt.imshow(srcimg) # 显示原图片

    # 对原图做 随机改变亮暗、对比度和颜色等 数据增强
    srcimg_gtbox = records[0]['gt_bbox']
    srcimg_label = records[0]['gt_class']

    plt.subplot(1,2,2)
    plt.title('Enhance Image', color='#0000FF')
    plt.axis('off') # 不显示坐标轴
    plt.imshow(img_enhance)

cname2cid = get_insect_names()
print("cname2cid:", cname2cid)
records = get_annotations(cname2cid, TRAINDIR)
print("len(records):", len(records))
print("records[0]:", records[0])


record = records[0]
img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
print("img.shape:", img.shape)
print("gt_boxes.shape:", gt_boxes.shape)
print("gt_labels:", gt_labels)
print("scales:", scales)


image_path = records[0]['im_file']
print("read image from file {}".format(image_path))
srcimg = Image.open(image_path)
# 将PIL读取的图像转换成array类型
srcimg = np.array(srcimg)

# 对原图做 随机改变亮暗、对比度和颜色等 数据增强
img_enhance = random_distort(srcimg)
visualize(srcimg, img_enhance)