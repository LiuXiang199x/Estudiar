from cv2 import add
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2 as cv

image = cv.imread("/home/agent/Estudiar/DeepLearning/ImgAugmenters/bedroom.jpg")


def add_blur(img_path, save_path):
    
    img2 = [img_path for _ in range(20)]
    
    # input img 2 seq with list
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.MotionBlur(k=20, angle=0),
            iaa.MotionBlur(k=20, angle=45),
            iaa.MotionBlur(k=20, angle=90),
            
            iaa.LinearContrast(0.5),
            iaa.Multiply(0.5,per_channel=0.5),
            iaa.Multiply(2,per_channel=0.5),
        ]),
    ])
    img_aug = seq(images=img2)
    
    # visulization
    ia.imshow(np.hstack(img_aug))
    for i in range(np.array(img_aug).shape[0]):
        img = img_aug[i]
        cv.imshow("qweqwe", img)
        cv.waitKey(0)
        
        # save img
        cv.imwrite(img, save_path)
        
add_blur(img_path=image)