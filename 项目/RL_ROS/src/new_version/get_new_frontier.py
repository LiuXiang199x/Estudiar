import numpy as np
import torch
"""
vector<vector<double>> get_frontier(vector<vector<double>> explored_map, 
	vector<vector<double>> occupancy_map, int row, int column) {

	// global map[0] = map occupancy: -1/100-->1(unexplored space/obstacles); 0-->0(free space) --- expand with 1
	// global map[1] = explored states: 0/100-->1(free space/obstacles); -1-->0(unexplored space) --- expand with 0
	vector<vector<double>> map_frontier(1200, vector<double>(1200, 0));
	for (int i = 1; i < 1199; i++) {
		for (int j = 1; j < 1199; j++) {
			double tmp = explored_map[i][j-1] + explored_map[i][j+1] + explored_map[i-1][j] + explored_map[i+1][j];
			if (explored_map[i][j] == 0 && tmp != 0) {
				map_frontier[i][j] = 1;
			}
		}
	}
	for (int i = 0; i < 1200; i++) {
		for (int j = 0; j < 1200; j++) {
			map_frontier[i][j] = map_frontier[i][j] * occupancy_map[i][j];
		}
	}
	return map_frontier;
}
"""

def get_frontiermap():
 # map occupancy: -1/100-->1(unexplored space/obstacles); 0-->0(free space)
 # explored states: 0/100-->1(free space/obstacles); -1-->0(unexplored space)
 print("====== Start testing ======")
 exploredmap = np.zeros((10,10))
 globalmap = np.ones((10,10))
 for i in range(3,7):
  for j in range(3,7):
   globalmap[i][j] = 0
   exploredmap[i][j] = 1
 print(globalmap)
 print(exploredmap)

 print("======> start getting frontiers <======")
 for i in range(len(globalmap)):
  for j in range(len(globalmap[i])):
   if globalmap[i][j] == 0:
    

def test_frontiermap():
 get_frontiermap()

if __name__ == "__main__":
 test_frontiermap()