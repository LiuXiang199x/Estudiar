from re import I
import torch
import cv2 as cv
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

image = cv.imread("/home/agent/Estudiar/DeepLearning/ImgAugmenters/bedroom.jpg")

def random_motion_blur():
    images = [image for _ in range(8)]
    # k = kernel size, angle = blur angle
    seq = iaa.Sequential([iaa.MotionBlur(k=10),
                        iaa.MotionBlur(k=10, angle=[-45,45]),
                        iaa.MotionBlur(k=10, angle=[-20,0])], random_order=True)

    img_aug = seq(images=images)
    ia.imshow(np.hstack(img_aug))
    # print(np.array(img_aug).shape)
    for i in range(np.array(img_aug).shape[0]):
        img = img_aug[i]
        cv.imshow("12", img)
        cv.waitKey(0)

def motion_blur_test():
    # k = kernel size, angle = blur angle
    seq = iaa.Sequential([iaa.MotionBlur(k=90, angle=90)])

    img_aug = seq(images=[image])
    ia.imshow(np.hstack(img_aug))

def motion_blur_shift():
    # k = kernel size, angle = blur angle
    seq = iaa.Sequential([iaa.Mo(k=90, angle=90)])

    img_aug = seq(images=[image])
    ia.imshow(np.hstack(img_aug))
    
motion_blur_shift()