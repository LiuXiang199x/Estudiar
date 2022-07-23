import numpy as np

testInput = np.array([[0,0,0,0,0,0],
                                            [0,0,1,0,0,2],
                                            [3,0,0,0,4,0],
                                            [0,9,5,7,0,11],
                                            [12,13,1,15,16,17],
                                            [0,20,4,0,9,0]])


def findclosetpth(maze, mazeGray):
    rowIndex, colIndex = np.where(maze==1)

    result = {}

    for i in range(len(rowIndex)):
        num = recursivePth(mazeGray, rowIndex[i], colIndex[i])   
        result["{}, {}".format(rowIndex[i], colIndex[i])] = num
    #     print("====================================")
    # print(result)
    return result


def recursivePth(maze, row, col):
    # print("row:{}, col:{}".format(row, col))

    flags = True
    tmpNum = 0
    resNum = 0
    # 越界问题
    for _ in range(maze.shape[0]):
        for tmpRow in range(row-tmpNum, row+tmpNum+1):
            if flags:
                for tmpCol in range(col-tmpNum, col+tmpNum+1):
                    # print(maze[tmpRow][tmpCol])
                    # print(tmpRow, tmpCol)
                    if tmpRow<0 or tmpRow>= maze.shape[0] or tmpCol<0 or tmpCol>=maze.shape[1]:
                        resNum=0
                    else:
                        if maze[tmpRow, tmpCol] != 0 and maze[tmpRow, tmpCol] != 1:
                            resNum = maze[tmpRow, tmpCol]
                            flags = False
                            break
            if not flags:
                break
        
        tmpNum += 1
        if not flags:
                break
    return resNum
    

# findclosetpth(testInput)

    