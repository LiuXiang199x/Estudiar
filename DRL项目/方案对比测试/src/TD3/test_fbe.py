import numpy as np

def DeterPoint(map, row, column):

    for i in [row - 1, row, row + 1]:
        for j in [column - 1, column, column + 1]:
            if map[i][j] == -1:
                return True
    return False

def FBE(map, row, column, mark):

    for i in [row - 1, row, row + 1]:
        for j in [column - 1, column, column + 1]:
            if map[i][j] == 0 and DeterPoint(map, i, j):
                map[i][j] = mark
                map = FBE(map, i, j, mark)
    return map

mark = -2
frontier_localmap = np.random.randint(0, 3, (800, 800)) - 1
frontier_localmap[0:10, :] = 1
frontier_localmap[-11:-1, :] = 1
frontier_localmap[:, 0:10] = 1
frontier_localmap[:, -11:-1] = 1

for row in range(len(frontier_localmap)-1):
    for column in range(len(frontier_localmap[0])-1):
        if frontier_localmap[row][column] == 0 and DeterPoint(frontier_localmap, row, column):
            frontier_localmap[row][column] = mark
            frontier_localmap = FBE(frontier_localmap, row, column, mark)
            mark -= 1

print(frontier_localmap)