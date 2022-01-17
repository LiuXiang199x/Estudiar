# coding:utf-8
import cv2 as cv
import torch
import numpy as np

src = cv.imread('Image/opencv.jpeg')
# imread() // imshow() // waitKey() // destroyAllWindows() // imwrite()
# cv.split() // cv.merge() // cv.cvtColor(xxx, cv.COLOR_BGR2XXX)

def load_save():
    ###### imread / namedWindow()/ imshow /
    #####  waitKey(0) / destroyAllWindows() / imwrite()
    print(cv.__version__)
    src = cv.imread('Image/opencv.jpeg')
    # src = cv.imread('Image/test.png')

    print(src.shape)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    '''
    # cv.imshow("11", src_gray)

    b = cv.waitKey(0)    # return 13 # 按下不同按钮返回不同值
    c = cv.destroyAllWindows()   # return None
    cv.imwrite("Image/opencv_gray.jpeg", src_gray)
    '''

    print(src[:, :, 0].shape)
    img_new = np.random.randn(src.shape[0], src.shape[1], 3)
    print(img_new.shape)

    img_new[:, :, 0]  = src[:, :, 0]
    img_new[:, :, 1]  = src[:, :, 1]
    img_new[:, :, 2]  = src[:, :, 2]
    img_new = src

def RGB2XXX():
    # cv.imshow("src", src)
    # cv.waitKey(0)
    # HSV(Hue, Saturation, Value)：色调（H），饱和度（S），亮度（V）
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算
    # 红色为0°，绿色为120°,蓝色为240°。它们的补色是：黄色为60°，青色为180°,品红为300°；
    # 饱和度S：取值范围为0.0～1.0（0-255）；
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)（0-255）。

    b, g, r = cv.split(src)
    print(b.shape)
    print(src[:, :, 2].shape)
    ooo = 0
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i][j] == src[i][j][0]:
                ooo += 1  
    print(ooo)
    print(378*428)
    src[:, :, 2] = 0
    src_merge = cv.merge([b, g, r])
    
    cv.imshow("src", src_merge)
    cv.waitKey(0)

    # 可以看到  图片中的三通道 不是RGB而是 B G R

RGB2XXX()