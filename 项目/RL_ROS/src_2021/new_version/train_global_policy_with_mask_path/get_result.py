#!/usr/bin/env python
# coding:utf-8


from std_msgs.msg import String
from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseActionGoal
import numpy as np
import rospy
import csv
 

class Visualresults():

	def __init__(self):

		rospy.Subscriber("/robot_1/move_base_node/NavfnROS/plan", Path, self.path_data)
		rospy.Subscriber("/robot_1/move_base/goal", MoveBaseActionGoal, self.goal_data)
		rospy.Subscriber("/robot_1/map", OccupancyGrid, self.map_data)
		
		self.map_data = []
		self.path_data = []
		self.flag = ''
		self.get_goal_map = False
		self.get_goal_path = False


	def goal_data(self, data):
		self.get_goal_map = True
		self.get_goal_path = True

	def map_data(self, data):
		if self.get_goal_map:		
			self.map_data.append(len(data.data) - np.sum(np.array(data.data)==-1))		
			self.get_goal_map = False

	def path_data(self, data):
		if self.get_goal_path:
			self.path_data.append(len(data.poses))
			self.get_goal_path = False
		print(self.map_data)
		print(self.path_data)
	'''	
	def save2csv(self, mapdata, pathdata):

		mapdata = numpy.array(mapdata)*0.05*0.05
		pathdata = numpy.array(pathdata)*0.05
		with open('home/chenxr/桌面/mapdata.csv', 'w') as mapcsvfile:
			mapwriter  = csv.writer(mapcsvfile)
			for mapitem in mapdata:
				mapwriter.writerow(mapitem)
		with open('home/chenxr/桌面/pathdata.csv', 'w') as pathcsvfile:
			pathwriter  = csv.writer(pathcsvfile)
			for pathitem in pathdata:
				pathwriter.writerow(pathitem)
	'''

if __name__ == '__main__':
	rospy.init_node("visual_testing_results", anonymous=True, log_level=rospy.DEBUG)
	a=Visualresults()
	rospy.spin()
