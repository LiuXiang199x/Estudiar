import cv2 as cv

img = cv.imread("Image/opencv.jpeg")

b, g, r = cv.split(img)
img[:, :, 0] = 0
img[:, :, 1] = 0
img[:, :, 2] = 0
cv.imshow("src", img)
out = cv.waitKey(0)
print(r.shape)
print(type(r))
