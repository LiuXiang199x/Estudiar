
# Average hashing

import cv2
import numpy as np
import imagehash

def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_sum = np.sum(img)
    img_mean = img_sum / 64
    img_finger = np.where(img > img_mean, 1, 0)
    return img_finger
   
if __name__ == '__main__':
    img1 = cv2.imread('DeepLearning/CNN/Project/图片去重/test_img2/2.png')
    img1_phash = aHash(img1)
    
    img2 = cv2.imread('DeepLearning/CNN/Project/图片去重/test_img2/3.png')
    img2_phash = aHash(img2)
    
    print("img1_phash:\n", img1_phash)
    print("img2_phash:\n", img2_phash)
    
    isquel = img1_phash == img2_phash
    print("isquel:\n", isquel)
    index = isquel == True
    print("index of True:\n", index)
    
    han = isquel[index]
    print("han:\n", han)
    
    #两张图片的相似度
    Nothanming = len(han)
    print(Nothanming)
    