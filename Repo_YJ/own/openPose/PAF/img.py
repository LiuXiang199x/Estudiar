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


joint_distance = np.linalg.norm(joint_to - joint_from)
unit_vector = (joint_to - joint_from) / joint_distance
rad = np.pi / 2
rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
# print("垂直分量 = ", np.dot(rot_matrix,(joint_to - joint_from) ))
vertical_unit_vector = np.dot(rot_matrix, unit_vector)  # 垂直分量
print("vertical_unit_vector = ", vertical_unit_vector)
grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()  # grid_x, grid_y用来遍历图上的每一个点

horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)
plt.imshow(horizontal_paf_flag)
