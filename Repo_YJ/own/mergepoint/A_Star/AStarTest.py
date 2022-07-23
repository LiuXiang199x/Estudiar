# coding=utf-8
import map2d
import AStar

if __name__ == '__main__':
    ##构建地图
    mapTest = map2d.map2d();
    mapTest.showMap();
    ##构建A*
    aStar = AStar.AStar(mapTest, AStar.Node(AStar.Point(1,1)), AStar.Node(AStar.Point(4,18)))
    print("A* start:")
    ##开始寻路
    if aStar.start():
        aStar.setMap();
        mapTest.showMap();
    else:
        print("no way")