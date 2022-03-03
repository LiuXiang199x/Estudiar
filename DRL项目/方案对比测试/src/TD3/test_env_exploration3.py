import rospy
import subprocess
import sys
sys.setrecursionlimit(100000)
from os import path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from numpy import inf
import numpy as np
import random
import math
from gazebo_msgs.msg import ModelState
from squaternion import Quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import os
import time

from nav_msgs.msg import OccupancyGrid

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Twist
from math import pi
import torch
#from concurrent.futures import ThreadPoolExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
def process_process_frontier(unexplored_mask):
    """
    Inputs:
        maps - (bs, 2, M, M) --- 1st channel is prob of obstacle present
                             --- 2nd channel is prob of being explored
    """
    G = unexplored_mask.shape[1]
    unexplored_mask_exp = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            if unexplored_mask[i, j]:
                unexplored_mask_exp[i, j] = True
                if i >= 1:
                    unexplored_mask_exp[i - 1, j] = True
                if i <= (G - 2):
                    unexplored_mask_exp[i + 1, j] = True
                if j >= 1:
                    unexplored_mask_exp[i, j - 1] = True
                if j <= (G - 2):
                    unexplored_mask_exp[i, j + 1] = True
    return unexplored_mask_exp

def _process_maps_frontier(maps):
    """
    Inputs:
        maps - (bs, 2, M, M) --- 1st channel is prob of obstacle present
                             --- 2nd channel is prob of being explored
    """
    bs = maps.shape[0]
    thresh_obstacle = 0.6
    thresh_explored = 0.6
    unexplored_mask = maps[:, 1] <= thresh_explored  # (bs, G, G)

    free_mask = (maps[:, 0] <= thresh_obstacle) & (
            maps[:, 1] > thresh_explored
    )  # (bs, G, G)

    device = free_mask.device
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.map(process_process_frontier, unexplored_mask.cpu())
    unexplored_mask_exp = torch.Tensor(list(result)) == 1

    frontier_mask = free_mask & unexplored_mask_exp.to(device)  # (bs, G, G) #bool
    #action_mask = frontier_mask.view(bs, -1)  # (bs, G*G)
    frontier_map = frontier_mask.type(torch.float32)
    return frontier_map
'''

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goalOK = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goalOK = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goalOK = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goalOK = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goalOK = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goalOK = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goalOK = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goalOK = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goalOK = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goalOK = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goalOK = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goalOK = False
    return goalOK

# Function to put the laser data in bins
def binning(lower_bound, data, quantity):
    width = round(len(data) / quantity)
    quantity -= 1
    bins = []
    for low in range(lower_bound, lower_bound + quantity * width + 1, width):
        bins.append(min(data[low:low + width]))
    return np.array([bins])

def DeterPoint(map, row, column):
    """
    for i in [row - 1, row, row + 1]:
        for j in [column - 1, column, column + 1]:
            if map[i][j] == -1:
                return True
    return False
    """
    if map[row-1][column] == -1 or map[row+1][column] == -1 or map[row][column-1] == -1 or map[row][column+1] == -1:
        return True
    return False

def FBE(map, row, column, mark):

    for i in [row - 1, row, row + 1]:
        for j in [column - 1, column, column + 1]:
            if map[i][j] == 0 and DeterPoint(map, i, j):
                map[i][j] = mark
                map = FBE(map, i, j, mark)
    return map

class GazeboEnv:
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile, height, width, nchannels):

        self.__robot_size = 0.455 // 0.05

        self.odomX = 0
        self.odomY = 0


        self.goalX = 1
        self.goalY = 0.0

        self.upper = 1.6  # 5.0
        self.lower = -1.6  # -5.0
        self.velodyne_data = np.ones(20) * 10

        self.set_self_state = ModelState()
        self.set_self_state.model_name = 'r1'
        self.set_self_state.pose.position.x = 0.
        self.set_self_state.pose.position.y = 0.
        self.set_self_state.pose.position.z = 0.
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]]
        for m in range(19):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03

        port = '11311'
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        time.sleep(2.0)

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True)

        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")
        self.fullpath = fullpath
        self.episode_num = 0
        self.gz_Process = subprocess.Popen(["roslaunch", "-p", port, fullpath, "gz_world:=map_6.world"])  # house_world.world"
        print("Gazebo launched!")
        # self.gzclient_pid = 0

        topic = 'vis_mark_array'
        self.publisher = rospy.Publisher(topic, MarkerArray, queue_size=3)

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.quaternions = list()
        euler_angles = (pi / 2, pi, 3 * pi / 2, 0)
        for angle in euler_angles:
            q = Quaternion.from_euler(0, 0, angle)
            self.quaternions.append(q)
        print(self.quaternions)
        print("env created")
        print("(self.goalX, self.goalY)", self.goalX, self.goalY)
        print("(self.odomX, self.odomY)", self.odomX, self.odomY)
        self.M = 1201 # 481
        self.visited_map = torch.zeros(1, 1, self.M, self.M)
        self.visited_map_OneStep = torch.zeros(1, self.M, self.M)
        rospy.Subscriber("/odom", Odometry, self._pose_callback)

    def MapsCb(self, globalmapCb):
        frontier_localmap = np.reshape(np.array(globalmapCb.data), (globalmapCb.info.height, globalmapCb.info.width))
        frontier_localmap = frontier_localmap.astype(np.int16)
        frontier_map = np.full((globalmapCb.info.height, globalmapCb.info.width), 0, dtype=int)
        mark = -2
        for row in range(len(frontier_localmap)):
            for column in range(len(frontier_localmap[0])):
                if frontier_localmap[row][column] == 0 and DeterPoint(frontier_localmap, row, column):
                    frontier_localmap[row][column] = mark
                    frontier_localmap = FBE(frontier_localmap, row, column, mark)
                    mark -= 1
        for i in range(len(frontier_localmap)):
            for j in range(len(frontier_localmap[0])):
                if frontier_localmap[i][j] <= -2:
                    frontier_map[i][j] = 1
        score = {}
        score_small = []
        globalcosmap_1 = []
        for a in frontier_localmap:
            globalcosmap_1 += set(a)
        globalcosmap_1 = set(globalcosmap_1)

        # filter some short frontier
        for item in set(globalcosmap_1):
            # print("[%d]:%d"%(item, np.sum(np.array(globalcosmap)==item)))
            if item < -1 and np.sum(np.array(frontier_localmap) == item) > self.__robot_size:
                score[item] = np.sum(np.array(frontier_localmap) == item)
            elif item < -1 and np.sum(np.array(frontier_localmap) == item) <= self.__robot_size:
                score_small.append(item)
            # print(np.sum(np.array(frontier_localmap)==item))
        for i in range(len(frontier_localmap)):
            for j in range(len(frontier_localmap[i])):
                if frontier_localmap[i][j] in score_small:
                    frontier_localmap[i][j] = -1
                    frontier_map[i][j] = 0

        global_frontier_map = torch.zeros(1, self.M, self.M)
        origin_x = globalmapCb.info.origin.position.x
        origin_y = globalmapCb.info.origin.position.y
        map_H = int(globalmapCb.info.height)
        map_W = int(globalmapCb.info.width)

        origin_x, origin_y = self.world2map(origin_x, origin_y, self.M, self.M, 0.05)
        H_d = int(origin_x)
        W_d = int(origin_y)
        for i in range(map_H):
            for j in range(map_W):
                if self.M > i + H_d >= 0 and self.M > j + W_d >= 0:
                    if frontier_map[i][j] == 1:  # unexplored space
                        global_frontier_map[0][i + H_d][j + W_d] = 1

        return global_frontier_map

    def _pose_callback(self, data):
        pose_x = data.pose.pose.position.x
        pose_y = data.pose.pose.position.y
        robot_at_map_x, robot_at_map_y = self.world2map(pose_x, pose_y, self.M, self.M, 0.05)
        self.visited_map[0, 0, robot_at_map_x, robot_at_map_y] = 1.
        self.visited_map_OneStep[0, robot_at_map_x, robot_at_map_y] = 1.

    def world2map(self, x_world, y_world, H, W, map_scale):
        """
        Here we convert world coordinates to map coordinates
        """
        Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
        Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
        #x_map = int(Hby2 - y_world / map_scale)
        #y_map = int(Wby2 + x_world / map_scale)
        x_map = int(Hby2 + y_world / map_scale)
        y_map = int(Wby2 + x_world / map_scale)
        return x_map, y_map

    def _map_to_global(self, map):
        global_map = torch.zeros(1, 2, self.M, self.M)
        global_map[:, 0, :, :] = 1
        global_map[:, 1, :, :] = 0

        origin_x = map.info.origin.position.x
        origin_y = map.info.origin.position.y
        map_H = int(map.info.height)
        map_W = int(map.info.width)

        map_data = np.array(map.data)

        map_data_ = np.reshape(map_data, (map_H, map_W))
        origin_x, origin_y = self.world2map(origin_x, origin_y, self.M, self.M, 0.05)
        H_d = int(origin_x)
        W_d = int(origin_y)
        for i in range(map_H):
            for j in range(map_W):
                if self.M > i + H_d >= 0 and self.M > j + W_d >= 0:
                    if map_data_[i][j] == -1:  # unexplored space
                        global_map[0][0][i + H_d][j + W_d] = 1
                        global_map[0][1][i + H_d][j + W_d] = 0
                    elif map_data_[i][j] == 0:  # free space
                        global_map[0][0][i + H_d][j + W_d] = 0
                        global_map[0][1][i + H_d][j + W_d] = 1
                    elif map_data_[i][j] == 100:  # obstacles
                        global_map[0][0][i + H_d][j + W_d] = 1
                        global_map[0][1][i + H_d][j + W_d] = 1

        return global_map

    def get_costmap(self):
        print("waiting for costmap")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/move_base/local_costmap/costmap', OccupancyGrid, timeout=0.5)
            except:
                # print("NO MAP")
                # sl_Process.terminate()
                pass
        # print("data:", data)
        map_data = np.array(data.data)
        # print("bf_reshape:", map_data.shape)
        # print("data.info:", data.info)
        map_data = map_data.reshape((data.info.width, data.info.height))
        # print("af_reshape:", map_data.shape)

        print("obtained costmap")

        return map_data

    def get_map(self):
        print("waiting for map")
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/map', OccupancyGrid, timeout=0.5)
            except:
                pass
        '''
        map_data = np.array(data.data)
        print("bf_reshape:", map_data.shape)
        print("data.info:", data.info)
        map_data = map_data.reshape((data.info.width, data.info.height))
        print("af_reshape:", map_data.shape)
        '''
        return data

    def step(self, act, sl_Process):

        self.goalX = act[0]
        self.goalY = act[1]
        """
        self.change_goal()
        """
        sl_Process.terminate()
        time.sleep(0.5)
        port = '11311'
        sl_Process = subprocess.Popen(
            ["roslaunch", "-p", port, os.path.join(os.path.dirname(__file__), "assets", "test_move_base7.launch")])
        time.sleep(4.0)

        goal_init = MoveBaseGoal()
        goal_init.target_pose.header.frame_id = 'map'
        goal_init.target_pose.header.stamp = rospy.Time.now()
        goal_init.target_pose.pose = Pose(Point(self.goalX, self.goalY, 0.0), self.quaternions[0])

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        move_base.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Connected to move base server")
        rospy.loginfo("Starting navigation test")

        self.visited_map_OneStep.fill_(0)

        move_base.send_goal(goal_init)
        finished_within_time = False
        finished_within_time = move_base.wait_for_result(rospy.Duration(100))
        if not finished_within_time:
            move_base.cancel_goal()
            rospy.loginfo("Timed out achieving goal")
        else:
            state = move_base.get_state()
            rospy.loginfo(state)
            if state == 3:
                rospy.loginfo("Goal succeeded!")
            else:
                rospy.loginfo("Goal Un-achieved!")

        dataOdom = None
        while dataOdom is None:
            try:
                dataOdom = rospy.wait_for_message('/odom', Odometry, timeout=0.5)
            except:
                pass

        map = self.get_map()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pass
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # Calculate robot heading from odometry data
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y

        state = np.array([self.odomX, self.odomY])

        M = self.M
        global_pose = torch.zeros(1, 3)
        global_pose[0, 0] = state[0]
        global_pose[0, 1] = state[1]
        #global_map = torch.zeros(1, 2, M, M)    #  use a function to process map
        global_map = self._map_to_global(map)
        collision_map = torch.zeros(1, M, M)
        visited_map = self.visited_map
        #frontier_map = torch.ones(1, M, M)
        #frontier_map = _process_maps_frontier(global_map)
        frontier_map = self.MapsCb(map)   #(1, M, M)
        path_length = torch.sum(self.visited_map_OneStep, dim=(1, 2))

        observations = {
            "global_pose": global_pose,
            "global_map": global_map,
            "collision_map": collision_map,
            "visited_map": visited_map,
            "frontier_map": frontier_map,
            "path_length": path_length,
        }

        return observations, sl_Process

    def reset(self, sl_Process):

        sl_Process.terminate()
        time.sleep(1.0)

        self.gz_Process.terminate()
        time.sleep(3.0)

        port = '11311'
        self.episode_num = 1 #4 #+= 1
        if self.episode_num > 12:
            self.episode_num = 0

        x = 0
        y = 0

        '''
        chk = False
        while not chk:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            chk = check_pos(x, y)
        '''

        self.gz_Process = subprocess.Popen(["roslaunch", "-p", port, self.fullpath, f"gz_world:=map_{self.episode_num}.world", f"x_init:={x}", f"y_init:={y}"])
        print("Gazebo Reset!")
        time.sleep(4.0)

        self.change_goal()

        sl_Process_new = subprocess.Popen(
            ["roslaunch", "-p", port, os.path.join(os.path.dirname(__file__), "assets", "test_move_base7.launch")])
        time.sleep(4)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        dataOdom = None
        while dataOdom is None:
            try:
                dataOdom = rospy.wait_for_message('/odom', Odometry, timeout=0.5)
            except:
                pass

        map = self.get_map()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pass
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # Calculate robot heading from odometry data
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y

        state = np.array([self.odomX, self.odomY])

        M = self.M
        global_pose = torch.zeros(1, 3)

        global_pose[0, 0] = state[0]
        global_pose[0, 1] = state[1]
        #global_map = torch.zeros(1, 2, M, M)    #  use a function to process map
        global_map = self._map_to_global(map)
        collision_map = torch.zeros(1, M, M)
        #visited_map = torch.zeros(1, 1, M, M)
        visited_map = self.visited_map
        #frontier_map = torch.ones(1, M, M)
        #frontier_map = _process_maps_frontier(global_map)
        frontier_map = self.MapsCb(map)
        path_length = torch.zeros(1, 1)
        observations = {
            "global_pose": global_pose,
            "global_map": global_map,
            "collision_map": collision_map,
            "visited_map": visited_map,
            "frontier_map": frontier_map,
            "path_length": path_length,
        }

        return observations, sl_Process_new

    # Place a new goal and check if its lov\cation is not on one of the obstacles
    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        gOK = False

        while not gOK:
            self.goalX = self.odomX + random.uniform(self.upper, self.lower)
            self.goalY = self.odomY + random.uniform(self.upper, self.lower)
            gOK = check_pos(self.goalX, self.goalY)
        print("goal changed", self.goalX, self.goalY)

    def move(self, sl_Process):

        '''
        goal_init.target_pose.pose.position.x = self.goalX
        goal_init.target_pose.pose.position.y = self.goalY
        goal_init.target_pose.pose.position.z = 0.0
        goal_init.target_pose.pose.orientation.x = 0.0
        goal_init.target_pose.pose.orientation.y = 0.0
        goal_init.target_pose.pose.orientation.z = 0.0
        goal_init.target_pose.pose.orientation.w = 1.0
        '''
        '''
        rospy.loginfo(goal_init.target_pose.pose)
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        '''
        print("=========Goal==============", self.goalX, self.goalY)
        goal_init = MoveBaseGoal()
        goal_init.target_pose.header.frame_id = 'map'
        goal_init.target_pose.header.stamp = rospy.Time.now()
        goal_init.target_pose.pose = Pose(Point(self.goalX, self.goalY, 0.0), self.quaternions[0])
        sl_Process.terminate()

        port = '11311'

        sl_Process = subprocess.Popen(
            ["roslaunch", "-p", port, os.path.join(os.path.dirname(__file__), "assets", "test_move_base7.launch")])

        move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")

        move_base.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Connected to move base server")
        move_base.send_goal(goal_init)
        finished_within_time = False
        finished_within_time = move_base.wait_for_result(rospy.Duration(60))
        if not finished_within_time:
            move_base.cancel_goal()
            rospy.loginfo("Timed out achieving goal")
            #sl_Process.terminate()
            return False
        else:
            state = move_base.get_state()
            rospy.loginfo(state)
            if state == 3:
                rospy.loginfo("Goal succeeded!")
                #sl_Process.terminate()
                return True
            else:
                rospy.loginfo("Goal Un-achieved!")
                #sl_Process.terminate()
                return False
