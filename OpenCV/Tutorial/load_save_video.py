from cgi import test
import cv2 as cv
import numpy as py

test_video = "Video/test.mp4"

#### cv.VideoCapture --> cap, cap.read()-->ret,frame
# cv.split() // cv.merge() // cv.capture() // ret, frame = cv.read()

cap = cv.VideoCapture(test_video)    # cap --- class
while 1:
    ret,frame = cap.read()   # ret__bool
    if ret == False:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("video", frame_gray)
    # waitKey(x): 等待x秒，如果在x秒期间，按下任意键，则立刻结束并返回按下键的ASCll码，否则返回-1
    # 若 x=0，那么会无限等待下去，直到有按键按下。
    out = cv.waitKey(40)
    print(out)
    if out == 27:    # 27 是esc的ASIIC码
        break