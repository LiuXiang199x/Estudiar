import cv2 as cv

"""
对lena 的脸提取ROI后修改再接回去：
    1.通过 [] 选择要提取的ROI部分
    2.对提取的ROI部分做处理。比如我们把它转化为灰度图
    3.想把修改后的ROI接回去，若是灰度图一定要先转为RGB图，因为通道数不一样，接回去会报错！！！
    4.gray 转换为 RGB
    5.RGB接回原图
"""

img_path = "./OpenCV/Image/opencv.jpeg"

def get_regular_roi():
    src_img = cv.imread(img_path)
    print(src_img.shape)
    src_cap = src_img[50:350, 50:350]
    src_cap_gray = cv.cvtColor(src_cap, cv.COLOR_BGR2GRAY)
    src_img[50:350, 50:350, 0] = src_cap_gray
    src_img[50:350, 50:350, 1] = src_cap_gray
    src_img[50:350, 50:350, 2] = src_cap_gray
    cv.imshow("ROI", src_img)
    cv.waitKey(0)

flower_img = "./OpenCV/Image/flower.png"
# maskkkkk
def get_iregular_roi(img_path_flower):
    # mask(遮罩)，OpenCV中是如此定义Mask的：八位单通道的Mat对象，
    # 每个像素点值为零或者非零区域。当Mask对象添加到图像区上时，
    # 只有非零的区域是可见，Mask中所有像素值为零与图像重叠的区域就会不可见
    # 通过mask 来提取不规则的ROI，需要调用的API是bitwise_and
    img = cv.imread(img_path_flower)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 通过cv.inRange 并查表，剥离出颜色
    mask = cv.inRange(img_hsv, (156,43,46),(100,255,255))

    cv.imshow("img:", mask)
    result = cv.bitwise_and(img, img, mask=mask)
    cv.imshow("result", result)
    cv.waitKey(0)

get_iregular_roi(flower_img)