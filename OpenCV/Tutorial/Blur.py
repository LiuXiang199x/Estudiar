import cv2 as cv
from cv2 import blur
from cv2 import imread
import numpy as np
from torch import float32

# 噪声在图像上常表现为一引起较强视觉效果的孤立像素点或像素块。
# 噪声的来源：
    # （1）图像获取过程中:采集图像过程中，由于受传感器材料属性、工作环境、
    # 电子元器件和电路结构等影响，会引入各种噪声，如电阻引起的热噪声、场效应管的沟道热噪声、光子噪声、暗电流噪声、光响应非均匀性噪声。
    # （2）图像信号传输过程中:由于传输介质和记录设备等的不完善，数字图像在其传输记录过程中往往会受到多种噪声的污染。

# 滤波Blur：是信号和图像处理中基本的任务。其目的是根据应用环境的不同，选择性的提取图像中某些认为是重要的信息。
# 过滤可以移除图像中的噪音、提取感兴趣的可视特征、允许图像重采样等等。

# 在频率分析领域的框架中，滤波器是一个用来增强图像中某个波段或频率并阻塞（或降低）其他频率波段的操作。
# 低通滤波器是消除图像中高频部分，但保留低频部分。高通滤波器消除低频部分。

# 高斯噪声 / 瑞里噪声 / 伽马噪声 / 指数分布噪声 / 均匀分布噪声 / 椒盐噪声

# 模糊操作的原理就是：卷积。  不同卷积核得到不同卷积效果。
# 均值：dst = cv.blur(image,(x,y))
# 中值：dst = cv.medianBlur(image,A)
# 自定义模糊：
# kernel = np.ones([5, 5], np.float32) / 25
# dst = cv.filter2D(src,depth,kernel,dst,another,delta,borderType)

img_path1 = "./OpenCV/Image/opencv.jpeg"
img_path2 = "./OpenCV/Image/test.jpg"
img_path3 = "./OpenCV/Image/opencv_gray.jpeg"
img_path4 = "./OpenCV/Image/flower.png"

img1 = cv.imread(img_path1)
img2 = cv.imread(img_path2)
img3 = cv.imread(img_path3)
img4 = cv.imread(img_path4)


# 均值
def blur_demo(img):
    dst = cv.blur(img, (20,20 ))
    cv.imshow("blur image:", dst)
    cv.waitKey(0)
    print(img.shape)
    print(dst.shape)

# 中值：处理椒盐噪声
def median_blur_demo(img):
    dst = cv.medianBlur(img, 5)
    cv.imshow("blur image:", dst)
    cv.waitKey(0)

# 自定义blur
def custom_blur_demo(img):
    # 均值kenrnel 和 cv.blur效果一样的
    kernel = np.ones([5,5], np.float32)/25

    # 拉普拉斯算子核
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    print(kernel)
    dst = cv.filter2D(img, -1, kernel)
    cv.imshow("img:", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

custom_blur_demo(img4)

# 拉普拉斯算子 —— 锐化 —— 边缘提取
# 拉普拉斯算子：[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]

# 高斯模型：GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
# 把高斯模型从一维扩展到二维
# 高斯双边（磨皮,非线性滤波）
