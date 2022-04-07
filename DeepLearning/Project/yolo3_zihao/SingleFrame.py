import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def look_img(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


net = cv2.dnn.readNet("/home/marco/yolo3.weights", "yolov3.cfg")

