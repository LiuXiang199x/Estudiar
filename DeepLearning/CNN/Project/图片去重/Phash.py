import cv2
import numpy as np
from itertools import chain

# Perceptual hashing
 
class PHash(object):
    @staticmethod
    def pHash(img_name):
        """
        get image pHash value
        """
        # 加载并调整图片为32x32灰度图片
        img = cv2.imread(img_name, 0)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
 
        # 创建二维列表
        h, w = img.shape[:2]
        vis0 = np.zeros((h, w), np.float32)
        vis0[:h, :w] = img  # 填充数据
 
        # 二维Dct变换
        vis1 = cv2.dct(cv2.dct(vis0))
        # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
        vis1.resize((32, 32), refcheck=False)
 
        # 把二维list变成一维list
        img_list = list(chain.from_iterable(vis1))
 
        # 计算均值
        avg = sum(img_list) * 1. / len(img_list)
        avg_list = ['0' if i < avg else '1' for i in img_list]
 
        # 得到哈希值，转成256位16进制数
        return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])
 
    @staticmethod
    def hammingDist(s1, s2):
        """
        计算两张图片的汉明距离
        """
        assert len(s1) == len(s2)
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
 
if __name__ == '__main__':
    HASH1 = PHash.pHash('DeepLearning/CNN/Project/图片去重/test_img2/2.png')
    HASH2 = PHash.pHash('DeepLearning/CNN/Project/图片去重/test_img2/2_2.png')
    # print(HASH1)
 
    #计算hanmming距离
    distance = PHash.hammingDist(HASH1, HASH2)
    #计算不一样的个数
    print('distance:',distance)
    out_score = 1 - distance * 1. / (32 * 32 / 4)
    #计算相似度
    print('similarity:',out_score)