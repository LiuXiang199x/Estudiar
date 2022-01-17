import cv2 as cv

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

img_path1 = "Image/opencv.jpeg"
img_path2 = "Image/test.jpg"
img1 = cv.imread(img_path1)
img2 = cv.imread(img_path2)
img3 = cv.imread("Image/opencv_gray.jpeg")

# cv.imshow("src", src)
# cv.waitKey(0)

def add(img1, img2):
    dat = cv.add(img1, img2)
    cv.imshow("addddd", dat)
    cv.waitKey(0)
    print(img1.shape)
    print(img2.shape)
    print("datttt::", dat.shape)


def subtract(img1, img2):
    dat = cv.subtract(img1, img2)
    cv.imshow("addddd", dat)
    cv.waitKey(0)
    print(img1.shape)
    print(img2.shape)
    print("datttt::", dat.shape)

def divide(img1, img2):
    dat = cv.divide(img1, img2)
    cv.imshow("addddd", dat)
    cv.waitKey(0)
    print(img1.shape)
    print(img2.shape)
    print("datttt::", dat.shape)

def multiply(img1, img2):
    dat = cv.multiply(img1, img2)
    cv.imshow("addddd", dat)
    cv.waitKey(0)
    print(img1.shape)
    print(img2.shape)
    print("datttt::", dat.shape)



subtract(img1, img3)