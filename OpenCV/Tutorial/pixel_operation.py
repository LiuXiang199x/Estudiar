import cv2 as cv
from torch import mul, std

# 像素的   逻辑运算，算术运算等等
# add / subtract / multiply / divide
# mean / stddev / meanStdDev 
# 若一个图片方差 = 0, 那么可以直接放弃这个图片了， 因为它根本不能称之为图片，因为没有任何信息，只能是一个常量表。
# 均值比较低，就说明它整体偏暗，因为值都偏小。
# 均值反映了图像的亮度，均值越大说明图像亮度越大，反之越小；
# 如果图片对比度小，那方差就小；如果图片对比度很大，那方差就大；

# 逻辑运算：与或非 cv.bitwise_and/not/or/xor (1 & 1 =1；1 & 0 = 1；0 & 0 = 0)
# and(AB) / or (A+B)不多解释了。not非(^A)，按位取反。xor异或=^AB+A^B

# cv.addWeighted()  给图片某些通道加上数值，增加亮度
# 阿斯顿萨
img_path1 = "./OpenCV/Image/opencv.jpeg"
img_path2 = "./OpenCV/Image/test.jpg"
img_path3 = "./OpenCV/Image/opencv_gray.jpeg"
img1 = cv.imread(img_path1)
img2 = cv.imread(img_path2)
img3 = cv.imread(img_path3)

# cv.imshow("src", src)
# cv.waitKey(0)

def add(img1, img2):
    dat = cv.add(img1, img2)
    cv.imshow("addddd", dat)
    cv.waitKey(0)
    print("img1.shape:", img1.shape)
    print("img1[0][0][0]:", img1[0][0][0])
    print("img2.shape:", img2.shape)
    print("img2[0][0][0]:", img2[0][0][0])
    print("datttt::", dat.shape)
    print("datttt[0][0][0]:", dat[0][0][0])


def subtract(img1, img2):
    dat = cv.subtract(img1, img2)
    cv.imshow("addddd", dat)
    cv.waitKey(0)
    print("img1.shape:", img1.shape)
    print("img1[0][0][0]:", img1[0][0][0])
    print("img2.shape:", img2.shape)
    print("img2[0][0][0]:", img2[0][0][0])
    print("datttt::", dat.shape)
    print("datttt[0][0][0]:", dat[0][0][0])

def divide(img1, img2):
    dat = cv.divide(img1, img2)
    cv.imshow("addddd", dat)
    cv.waitKey(0)
    print("img1.shape:", img1.shape)
    print("img1[0][0][0]:", img1[0][0][0])
    print("img2.shape:", img2.shape)
    print("img2[0][0][0]:", img2[0][0][0])
    print("datttt::", dat.shape)
    print("datttt[0][0][0]:", dat[0][0][0])

def multiply(img1, img2):
    dat = cv.multiply(img1, img2)
    cv.imshow("addddd", dat)
    cv.waitKey(0)
    print("img1.shape:", img1.shape)
    print("img1[0][0][0]:", img1[0][0][0])
    print("img2.shape:", img2.shape)
    print("img2[0][0][0]:", img2[0][0][0])
    print("datttt::", dat.shape)
    print("datttt[0][0][0]:", dat[0][0][0])


def mean_stddev_meanStdDev(img1, img2):
    mean_img1 = cv.mean(img1)
    mean_img2 = cv.mean(img2)

    a, b  = cv.meanStdDev(img1)  # return (array, array)
    stddev_img2 = cv.meanStdDev(img2)

    print("mean_img1:", mean_img1)
    print("stddev_img2:", a, b)

    cv.imshow("addddd", mean_img1)
    cv.waitKey(0)


def bitwise(img1, img2):
    dat_and = cv.bitwise_and(img1, img2)
    dat_or = cv.bitwise_or(img1, img2)
    dat_not = cv.bitwise_not(img1, img2)
    dat_xo = cv.bitwise_xor(img1, img2)

    cv.imshow("addddd", dat_not)
    cv.waitKey(0)

    print("img1.shape:", img1.shape)
    print("img1[0][0][0]:", img1[0][0][0])
    print("img2.shape:", img2.shape)
    print("img2[0][0][0]:", img2[0][0][0])
    print("datttt::", dat_not.shape)
    print("datttt[0][0][0]:", dat_not[0][0][0])

bitwise(img1, img3)