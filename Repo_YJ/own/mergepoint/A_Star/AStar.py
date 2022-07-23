# coding=utf-8

#描述AStar算法中的节点数据 
class Point:
    """docstring for point"""
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
        
class Node:     
    def __init__(self, point, g = 0, h = 0):  
        self.point = point        #自己的坐标  
        self.father = None        #父节点  
        self.g = g                #g值
        self.h = h                #h值  
  
    """
    估价公式：曼哈顿算法
     """
    def manhattan(self, endNode):
        self.h = (abs(endNode.point.x - self.point.x) + abs(endNode.point.y - self.point.y))*10 
    
    def setG(self, g):
        self.g = g

    def setFather(self, node):
        self.father = node

class AStar:
    """
    A* 算法 
    python 2.7 
    """
    def __init__(self, map2d, startNode, endNode):
        """ 
        map2d:      寻路数组 
        startNode:  寻路起点 
        endNode:    寻路终点 
        """  
        #开放列表
        self.openList = []
        #封闭列表  
        self.closeList = []
        #地图数据
        self.map2d = map2d
        #起点  
        self.startNode = startNode
        #终点
        self.endNode = endNode 
        #当前处理的节点
        self.currentNode = startNode
        #最后生成的路径
        self.pathlist = [];
        return;

    def getMinFNode(self):
        """ 
        获得openlist中F值最小的节点 
        """  
        nodeTemp = self.openList[0]  
        for node in self.openList:  
            if node.g + node.h < nodeTemp.g + nodeTemp.h:  
                nodeTemp = node  
        return nodeTemp

    def nodeInOpenlist(self,node):
        for nodeTmp in self.openList:  
            if nodeTmp.point.x == node.point.x \
            and nodeTmp.point.y == node.point.y:  
                return True  
        return False

    def nodeInCloselist(self,node):
        for nodeTmp in self.closeList:  
            if nodeTmp.point.x == node.point.x \
            and nodeTmp.point.y == node.point.y:  
                return True  
        return False

    def endNodeInOpenList(self):  
        for nodeTmp in self.openList:  
            if nodeTmp.point.x == self.endNode.point.x \
            and nodeTmp.point.y == self.endNode.point.y:  
                return True  
        return False

    def getNodeFromOpenList(self,node):  
        for nodeTmp in self.openList:  
            if nodeTmp.point.x == node.point.x \
            and nodeTmp.point.y == node.point.y:  
                return nodeTmp  
        return None

    def searchOneNode(self,node):
        """ 
        搜索一个节点
        x为是行坐标
        y为是列坐标
        """  
        #忽略障碍
        if self.map2d.isPass(node.point) != True:  
            return  
        #忽略封闭列表
        if self.nodeInCloselist(node):  
            return  
        #G值计算 
        if abs(node.point.x - self.currentNode.point.x) == 1 and abs(node.point.y - self.currentNode.point.y) == 1:  
            gTemp = 14  
        else:  
            gTemp = 10  


        #如果不再openList中，就加入openlist  
        if self.nodeInOpenlist(node) == False:
            node.setG(gTemp)
            #H值计算 
            node.manhattan(self.endNode);
            self.openList.append(node)
            node.father = self.currentNode
        #如果在openList中，判断currentNode到当前点的G是否更小
        #如果更小，就重新计算g值，并且改变father 
        else:
            nodeTmp = self.getNodeFromOpenList(node)
            if self.currentNode.g + gTemp < nodeTmp.g:
                nodeTmp.g = self.currentNode.g + gTemp  
                nodeTmp.father = self.currentNode  
        return;

    def searchNear(self):
        """ 
        搜索节点周围的点 
        按照八个方位搜索
        拐角处无法直接到达
        (x-1,y-1)(x-1,y)(x-1,y+1)
        (x  ,y-1)(x  ,y)(x  ,y+1)
        (x+1,y-1)(x+1,y)(x+1,y+1)
        """ 
        if self.map2d.isPass(Point(self.currentNode.point.x - 1, self.currentNode.point.y)) and \
        self.map2d.isPass(Point(self.currentNode.point.x, self.currentNode.point.y -1)):
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))
        
        self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y)))

        if self.map2d.isPass(Point(self.currentNode.point.x - 1, self.currentNode.point.y)) and \
        self.map2d.isPass(Point(self.currentNode.point.x, self.currentNode.point.y + 1)):
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))

        self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1)))
        self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1)))

        if self.map2d.isPass(Point(self.currentNode.point.x, self.currentNode.point.y - 1)) and \
        self.map2d.isPass(Point(self.currentNode.point.x + 1, self.currentNode.point.y)):
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y - 1)))
        
        self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y)))

        if self.map2d.isPass(Point(self.currentNode.point.x + 1, self.currentNode.point.y)) and \
        self.map2d.isPass(Point(self.currentNode.point.x, self.currentNode.point.y + 1)):
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y + 1)))
        return;

    def start(self):
        ''''' 
        开始寻路 
        '''
        #将初始节点加入开放列表
        self.startNode.manhattan(self.endNode);
        self.startNode.setG(0);
        self.openList.append(self.startNode)

        while True:
            #获取当前开放列表里F值最小的节点
            #并把它添加到封闭列表，从开发列表删除它
            self.currentNode = self.getMinFNode()
            self.closeList.append(self.currentNode)
            self.openList.remove(self.currentNode)

            self.searchNear();

            #检验是否结束
            if self.endNodeInOpenList():
                nodeTmp = self.getNodeFromOpenList(self.endNode)
                while True:
                    self.pathlist.append(nodeTmp);
                    if nodeTmp.father != None:
                        nodeTmp = nodeTmp.father
                    else:
                        return True;
            elif len(self.openList) == 0:
                return False;
        return True;

    def setMap(self):
        for node in self.pathlist:
            self.map2d.setMap(node.point);
        return;