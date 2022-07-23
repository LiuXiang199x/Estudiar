from copy import copy
from importlib import import_module
from xmlrpc.client import TRANSPORT_ERROR
from cv2 import cvtColor
import torch
import torchvision
import cv2 as cv
from PIL import Image
import numpy as np
import json
import copy
from findClosetPoint import *
from map2d import map2d
import AStar
import os

root_dir = "/home/marco/Estudiar/Repo_YJ/own/mergepoint/datas"
def divideRegion(visualizaton=False, visualizatonRegions=False):
    imgPth = "image/0.png"
    jsonPth = "/home/agent/Repo_YJ/project/source_datasets/conorPoint/标注/mergePoint/gaussianHeatmap/0.json"
    imgLabel = "/home/agent/Repo_YJ/project/source_datasets/conorPoint/标注/mergePoint/image/0_.png"

    visualizaton = visualizaton
    visualizatonRegions = visualizatonRegions

    img = cv.imread(imgPth)
    imgLabel = cv.imread(imgLabel)
    imgLabelGray = cv.cvtColor(imgLabel, cv.COLOR_RGB2GRAY)

    jsonFile = open(jsonPth).read()
    jsonContent = json.loads(jsonFile)
    key = [i for i in jsonContent.keys()]
    label_ = np.array(jsonContent[key[0]])
    labelBinary = label_==1

    if visualizaton:
        tmp_ = np.zeros((label_.shape[0], label_.shape[1], 3))
        tmp_[:, :, 0] = labelBinary * 255
        tmp_[:, :, 1] = labelBinary * 255
        tmp_[:, :, 2] = labelBinary * 255
        
        cv.imshow("imgBinary", tmp_)
        cv.imshow("imgRGB", img)
        cv.imshow("imgggg", imgLabel)
        cv.imshow("imgGray", imgLabelGray)
        cv.waitKey()
        cv.destroyAllWindows()

    # 0-1: labelBinary; imgLabelGray: search label;
    a = findclosetpth(labelBinary, imgLabelGray)
    labels = np.unique(np.array([i for i in a.values()]))
    s=[]
    for i in range(len(labels)):
        tmpp = []
        for j in a.keys():
            if a[j]==labels[i]:
                tmpp.append(j)
        s.append(tmpp)
    
    RegionsMap = []
    for item in s:
        tmpR = np.zeros((label_.shape[0], label_.shape[1]))
        for j in item:
            tmpR[int(j.split(",")[0]), int(j.split(",")[1])] = 1
        RegionsMap.append(tmpR)
    
    if visualizatonRegions:
        for room in s:
            tmp_ = np.zeros((label_.shape[0], label_.shape[1], 3))
            for j in room:
                # print(j.sqplit(","))
                tmp_[int(j.split(",")[0]), int(j.split(",")[1]), 0] = 255
                tmp_[int(j.split(",")[0]), int(j.split(",")[1]), 1] = 255
                tmp_[int(j.split(",")[0]), int(j.split(",")[1]), 2] = 255

            cv.imshow("imgBinary", tmp_)
            cv.waitKey()
            cv.destroyAllWindows()
    
    return RegionsMap

mapsList = np.array(divideRegion())
print(mapsList.shape)

if True:
    a = np.zeros((mapsList.shape[1], mapsList.shape[2], 3))
    a[:, :, 0] = mapsList[0]
    a[:, :, 1] = mapsList[0]
    a[:, :, 2] = mapsList[0]

    cv.imshow("asd", a)
    cv.waitKey()
    cv.destroyAllWindows()

tmap = np.array(mapsList[0])
pointS = np.where(tmap==1)
# tmap = [[1, 1, 1, 1, 1, 1],
#                 [1, 0, 0, 0, 0, 1],
#                 [1, 0, 0, 0, 0, 1],
#                 [1, 1, 0, 0, 1, 1],
#                 [1, 0, 0, 0, 0, 1]]

tt = map2d(np.array(tmap))
tt.showMap()
print(len(tt.data))
print(len(tt.data[0]))
print(np.array(tt.data).shape)
print(tmap.shape)

aStar = AStar.AStar(tt, AStar.Node(AStar.Point(pointS[0][0],pointS[1][0])), AStar.Node(AStar.Point(pointS[0][1],pointS[1][1])))
print("A* start:")
##开始寻路
if aStar.start():
    aStar.setMap();
    tt.showMap();
else:
    print("no way")

