import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2

#设置随机数种子
ia.seed(8)

def example():
    #读取图片
    example_img = cv2.imread("/home/agent/Estudiar/DeepLearning/ImgAugmenters/opencv.jpeg")
    #通道转换
    example_img = example_img[:, :, ::-1]
    #对图片进行缩放处理
    # example_img = cv2.resize(example_img,(224,224))
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        #随机裁剪图片边长比例的0~0.1
        iaa.Crop(percent=(0,0.1)),
        #Sometimes是指指针对50%的图片做处理

        #锐化处理
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

        #浮雕效果
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        
        iaa.Sometimes(
            0.5,
            #高斯模糊
            iaa.GaussianBlur(sigma=(0,0.5))
        ),
        #增强或减弱图片的对比度
        iaa.LinearContrast((0.75,1.5)),
        #添加高斯噪声
        #对于50%的图片,这个噪采样对于每个像素点指整张图片采用同一个值
        #剩下的50%的图片，对于通道进行采样(一张图片会有多个值)
        #改变像素点的颜色(不仅仅是亮度)
        iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),
        #让一些图片变的更亮,一些图片变得更暗
        #对20%的图片,针对通道进行处理
        #剩下的图片,针对图片进行处理
        iaa.Multiply((0.8,1.2),per_channel=0.2),
        #仿射变换
        
        # iaa.Affine(
        #     #缩放变换
        #     scale={"x":(0.8,1.2),"y":(0.8,1.2)},
        #     #平移变换
        #     translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)},
        #     #旋转
        #     rotate=(-25,25),
        #     #剪切
        #     shear=(-8,8)
        # )
        
    #使用随机组合上面的数据增强来处理图片
    ],random_order=True)
    #生成一个图片列表
    example_images = np.array(
        [example_img for _ in range(32)],
        dtype=np.uint8
    )
    aug_imgs = seq(images = example_images)
    #显示图片
    # ia.show_grid(aug_imgs,rows=4,cols=8)
    for i in range(aug_imgs.shape[0]):
        img = aug_imgs[i]
        # print(img.shape)(224, 224, 3)
        # cv2.imwrite("aug_%d.jpg"%i,img)
        cv2.imshow("img", img)
        cv2.waitKey(0)
example()

"""
#显示图片
    # ia.show_grid(aug_imgs,rows=4,cols=8)
    print(aug_imgs.shape)#(32, 224, 224, 3)
    for i in range(aug_imgs.shape[0]):
        img = aug_imgs[i]
        # print(img.shape)(224, 224, 3)
        cv2.imwrite("aug_%d.jpg"%i,img)
https://aistudio.baidu.com/aistudio/projectdetail/288691
"""