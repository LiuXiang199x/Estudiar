from multiprocessing.spawn import import_main_path
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2 as cv


image = cv.imread("/home/agent/Estudiar/DeepLearning/ImgAugmenters/bedroom.jpg")
sometimes = lambda aug: iaa.Sometimes(0.5, aug) #建立lambda表达式，

def operations():
    # k = kernel size, angle = blur angle
    seq = iaa.Sequential([
        #增强或减弱图片的对比度
        # iaa.LinearContrast(0.5),
        
        #让一些图片变的更亮,一些图片变得更暗
        #对20%的图片,针对通道进行处理
        #剩下的图片,针对图片进行处理
        iaa.Multiply((0.5,2),per_channel=0.5),
    ])
    img_aug = seq(images=[image])
    # ia.imshow(np.hstack(img_aug))
    for i in range(np.array(img_aug).shape[0]):
        img = img_aug[i]
        cv.imshow("qweqwe", img)
        cv.waitKey(0)
    
operations()