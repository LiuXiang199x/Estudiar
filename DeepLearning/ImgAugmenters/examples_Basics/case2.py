import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2

def simple_example():
    seq = iaa.Sequential([
        #从图片边随机裁剪50~100个像素,裁剪后图片的尺寸和之前不一致
        #通过设置keep_size为True可以保证裁剪后的图片和之前的一致
        iaa.Crop(px=(50,100),keep_size=False),
        #50%的概率水平翻转
        iaa.Fliplr(0.5),
        #50%的概率垂直翻转
        iaa.Flipud(0.5),
        #高斯模糊,使用高斯核的sigma取值范围在(0,3)之间
        #sigma的随机取值服从均匀分布
        iaa.GaussianBlur(sigma=(0,3.0))
    ])
    #可以内置的quokka图片,设置加载图片的大小
    # example_img = ia.quokka(size=(224,224))
    #这里我们使用自己的图片
    example_img = cv2.imread("/home/marco/Estudiar/DeepLearning/ImgAugmenters/opencv.jpeg")
    #对图片的通道进行转换,由BGR转为RGB
    #imgaug处理的图片数据是RGB通道
    example_img = example_img[:,:,::-1]
    #数据增强,针对单张图片
    aug_example_img = seq.augment_image(image=example_img)
    print(example_img.shape,aug_example_img.shape)
    #(700, 700, 3) (544, 523, 3)
    #显示图片
    ia.imshow(aug_example_img)

simple_example()
