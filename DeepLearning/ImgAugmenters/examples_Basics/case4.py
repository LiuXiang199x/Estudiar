from cv2 import exp
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib.pyplot import sca
import numpy as np
import cv2 as cv
import torch

example_img = cv.imread("/home/agent/Estudiar/DeepLearning/ImgAugmenters/opencv.jpeg")
example_img2 = cv.imread("/home/agent/Estudiar/DeepLearning/ImgAugmenters/bedroom.jpg")
images = [example_img]

seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(30, 90)),
    iaa.Crop(percent=(0, 0.4))], random_order=True)


images_aug = [seq(images=images) for _ in range(8)]

print("Augmented:")
ia.imshow(np.hstack(images_aug))