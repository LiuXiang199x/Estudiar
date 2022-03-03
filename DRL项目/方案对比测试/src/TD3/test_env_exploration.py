import rospy
import subprocess
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


class GazeboEnv:
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile, height, width, nchannels):

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
        self.gz_Process = subprocess.Popen(["roslaunch", "-p", port, fullpath, "gz_world:=house_world.world"])
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
        map_data = np.array(data.data)
        print("bf_reshape:", map_data.shape)
        print("data.info:", data.info)
        map_data = map_data.reshape((data.info.width, data.info.height))
        print("af_reshape:", map_data.shape)
        return map_data

    def step(self, act):
        target = False

        """
        self.goalX = act[0]
        self.goalY = act[1]
        """
        self.change_goal()
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        _ = self.move()

        dataOdom = None
        while dataOdom is None:
            try:
                dataOdom = rospy.wait_for_message('/odom', Odometry, timeout=0.5)
            except:
                pass

        map = self.get_costmap()

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

        return state, map

    def reset(self, sl_Process):

        sl_Process.terminate()
        time.sleep(1.0)

        self.gz_Process.terminate()
        time.sleep(0.5)
        port = '11311'
        self.episode_num += 1
        if self.episode_num > 2:
            self.episode_num = 1
        self.gz_Process = subprocess.Popen(
            ["roslaunch", "-p", port, self.fullpath, f"gz_world:=map_{self.episode_num}.world"])
        print("Gazebo Reset!")
        time.sleep(4.0)

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        chk = False
        while not chk:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            chk = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odomX = object_state.pose.position.x
        self.odomY = object_state.pose.position.y

        self.change_goal()
        # self.random_box()
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        sl_Process_new = subprocess.Popen(
            ["roslaunch", "-p", port, os.path.join(os.path.dirname(__file__), "assets", "test_move_base5.launch")])
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

        map = self.get_costmap()

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

        return state, map, sl_Process_new

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


    def move(self):

        goal_init = MoveBaseGoal()
        goal_init.target_pose.header.frame_id = 'map'
        goal_init.target_pose.header.stamp = rospy.Time.now()
        goal_init.target_pose.pose = Pose(Point(self.goalX, self.goalY, 0.0), self.quaternions[0])
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
        port = '11311'

        sl_Process = subprocess.Popen(
            ["roslaunch", "-p", port, os.path.join(os.path.dirname(__file__), "assets", "test_move_base5.launch")])

        move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        move_base.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Connected to move base server")
        rospy.loginfo("Starting navigation test")

        move_base.send_goal(goal_init)
        finished_within_time = False
        finished_within_time = move_base.wait_for_result(rospy.Duration(60))
        if not finished_within_time:
            move_base.cancel_goal()
            rospy.loginfo("Timed out achieving goal")
            sl_Process.terminate()
            return False
        else:
            state = move_base.get_state()
            rospy.loginfo(state)
            if state == 3:
                rospy.loginfo("Goal succeeded!")
                sl_Process.terminate()
                return True
            else:
                rospy.loginfo("Goal Un-achieved!")
                sl_Process.terminate()
                return False
