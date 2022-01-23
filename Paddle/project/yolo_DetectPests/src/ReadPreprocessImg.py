# 数据读取
import cv2
from ImagePathLabels import get_annotations, get_insect_names
import numpy as np
from PIL import Image, ImageEnhance
import random

TRAINDIR = '/home/marco/Datasets/Paddle/insects/train'
TESTDIR = '/home/marco/Datasets/Paddle/insects/test'
VALIDDIR = '/home/marco/Datasets/Paddle/insects/val'

def get_bbox(gt_bbox, gt_class):
    # 对于一般的检测任务来说，一张图片上往往会有多个目标物体
    # 设置参数MAX_NUM = 50， 即一张图片最多取50个真实框；如果真实
    # 框的数目少于50个，则将不足部分的gt_bbox, gt_class和gt_score的各项数值全设置为0
    MAX_NUM = 50
    gt_bbox2 = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]
        if i >= MAX_NUM:
            break
    return gt_bbox2, gt_class2

def get_img_data_from_file(record):
    """
    record is a dict as following,
      record = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
    """
    im_file = record['im_file']
    h = record['h']
    w = record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_bbox']
    difficult = record['difficult']

    img = cv2.imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # check if h and w in record equals that read from img
    assert img.shape[0] == int(h), \
             "image height of {} inconsistent in record({}) and img file({})".format(
               im_file, h, img.shape[0])

    assert img.shape[1] == int(w), \
             "image width of {} inconsistent in record({}) and img file({})".format(
               im_file, w, img.shape[1])

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)

    # gt_bbox 用相对值
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)
  
    return img, gt_boxes, gt_labels, (h, w)


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

# get_img_data_from_file()函数可以返回图片数据的数据，它们是图像数据img，
# 真实框坐标gt_boxes，真实框包含的物体类别gt_labels，图像尺寸scales。

