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
from map2d_int import map2d
import AStar


a = np.zeros((5,5))
q = map2d(a)
print(q.showMap())

aStar = AStar.AStar(a, AStar.Node(AStar.Point(0, 0)), AStar.Node(AStar.Point(3,3)))
print("A* start:")
##开始寻路
if aStar.start():
    aStar.setMap();
    q.showMap();
else:
    print("no way")