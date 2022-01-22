import cv2 as cv

img_path = "./OpenCV/Image/opencv.jpeg"
src_img = cv.imread(img_path)
print(src_img.shape)
cv.imshow()
# cv.waitKey(0)