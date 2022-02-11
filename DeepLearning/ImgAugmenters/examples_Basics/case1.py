from multiprocessing.spawn import import_main_path
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2 as cv


im = cv.imread("/home/agent/Estudiar/DeepLearning/ImgAugmenters/bedroom.jpg")
print(im.shape)
width, heigth = im.shape[:2]
# im = cv.resize(im, (224, 224)).astype(np.int8)
# print(im.shape)
images = np.zeros([2, width, heigth, 3])
print(images[0].shape)
images[0] = im
print(images.shape)
print(images[0]==im)
imm = images[0]
cv.imshow("img", imm)
cv.waitKey(0)

seq = iaa.Sequential([         #建立一个名为seq的实例，定义增强方法，用于增强
    iaa.Crop(px=(0, 16)),     #对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
    iaa.Fliplr(0.5),     #对百分之五十的图像进行做左右翻转
    iaa.GaussianBlur((0, 1.0))     #在模型上使用0均值1方差进行高斯模糊
])

images_aug = seq.augment_images(images)    #应用数据增强
print(images_aug.shape)

# cv.imshow("img", images_aug[0])
# cv.waitKey(0)