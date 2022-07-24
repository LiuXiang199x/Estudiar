#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""                  
*  * *** *  * *  *      
*  *  *   **  *  *             
****  *   **  *  *                 
*  *  *   **  *  *         
*  * **  *  * ****  
"""
import numpy as np
import matplotlib.pyplot as plt

paf_sigma = 8         # 肢体宽度
shape = (425, 640, 3) # 图像大小
joint_from = np.array([378, 118]) # 肢体的两个点
joint_to = np.array([393,  214])

plt.xlim((0,shape[1]))
plt.ylim((0,shape[0]))
plt.scatter([joint_from[0], joint_to[0]], [joint_from[1], joint_to[1]], color='b')
plt.gca().invert_yaxis() # 将plt的原点由坐下设置为左上
plt.show()