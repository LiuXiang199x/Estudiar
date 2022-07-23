# coding=utf-8
from __future__ import print_function
import numpy

class map2d:
    """ 
    地图数据
    """  
    def __init__(self, dataMatrix: numpy.array):
        
        self.data = dataMatrix
        self.w = self.data.shape[1]
        self.h = self.data.shape[0]

        self.passTag = 0
        self.pathTag = 1

    def showMap(self):
        for x in range(0, self.h):
            for y in range(0, self.w):
                print(self.data[x][y], end='')
            print(" ")
        return;

    def setMap(self, point):
        self.data[point.x][point.y] = self.pathTag
        return;

    def isPass(self, point):
        if (point.x < 0 or point.x > self.h - 1) or (point.y < 0 or point.y > self.w - 1):
            return False;

        if self.data[point.x][point.y] == self.passTag:
            return True;

# a = map2d()
# print(numpy.array(a.data[1]).shape)