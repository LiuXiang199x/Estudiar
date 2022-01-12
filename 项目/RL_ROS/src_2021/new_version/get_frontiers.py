import numpy as np
import pandas as pd
from collections import Counter

def deterpoint(map, idx, idy):
    if map[idx-1][idy] == -1:
        return True
    if map[idx+1][idy] == -1:
        return True
    if map[idx][idy-1] == -1:
        return True
    if map[idx][idy+1] == -1:
        return True
    return False

def filter_frontiers(map, idx, idy, min_size):
    row, column = map.shape
    # print("=====> row = {}; =====> column = {}".format(row, column))

    map_padding = np.zeros((row+min_size*2, column+min_size*2))
    print("map_padding.shape:", map_padding.shape)
    for i in range(len(map)):
        map_padding[min_size:min_size+row, min_size:min_size+column] = map
    print(map_padding)
    # print("======= map padding ========\n", map_padding)
    for i in range(1,min_size+1):
        if np.sum(map[idx-i-1:idx+i+1-1][idy-i-1:idy+i+1-1]==1) <= min_size and np.sum(map[idx-i:idx+i+1][idy-i:idy+i+1]==1) == np.sum(map[idx-i-1:idx+i+1-1][idy-i-1:idy+i+1-1]==1):
            return True





# MapOccupancy: 0 - free space; -1 - unexplored_space; 100 - obstacles
def get_frontiermap(mapOccup):
    minsize = 5
    print("======== start processing ========")
    frontier_map = np.zeros((len(mapOccup), len(mapOccup[0])))
    for i in range(len(mapOccup)):
        for j in range(len(mapOccup[0])):
            if mapOccup[i][j] == 0 and deterpoint(mapOccup, i, j):
                mapOccup[i][j] = 1

    print(mapOccup)
    for i in range(len(mapOccup)):
        for j in range(len(mapOccup[0])):
            if mapOccup[i][j] == 1 and filter_frontiers(mapOccup, i, j, minsize):
                frontier_map[i][j] = 0
            # print(frontier_map)

    print("========== frontier map ==========\n", frontier_map)


mapExplored = np.zeros((10, 10))
mapOccupancy = np.full((10, 10), -1)
for i in range(3,7):
    for j in range(3,7):
        mapOccupancy[i][j] = 0

get_frontiermap(mapOccupancy)