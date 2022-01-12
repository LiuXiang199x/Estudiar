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

        self.upper = 1.6 #5.0
        self.lower = -1.6 #-5.0
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

        gz_Process = subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")
        #self.gzclient_pid = 0

        topic = 'vis_mark_array'
        self.publisher = rospy.Publisher(topic, MarkerArray, queue_size=3)

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.quaternions = list() 
        euler_angles = (pi/2, pi, 3*pi/2, 0)
        for angle in euler_angles:
            q = Quaternion.from_euler(0, 0, angle)    
            self.quaternions.append(q)
        print(self.quaternions)
        print("env created")
        print("(self.goalX, self.goalY)", self.goalX, self.goalY)
        print("(self.odomX, self.odomY)", self.odomX, self.odomY)

    def get_map(self):
        #port = '11311'
        #sl_Process = subprocess.Popen(["roslaunch", "-p", port, "/media/agent/eb0d0016-e15f-4a25-8c28-0ad31789f3cb/ROS/DRL-robot-navigation/TD3/assets/test_move_base5.launch"])
        print("waiting for map")
        time.sleep(5)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/map', OccupancyGrid, timeout=0.5)
            except:
                #print("NO MAP")
                #sl_Process.terminate()
                pass

        map_data = np.array(data.data)
        print("bf_reshape:", map_data.shape)
        print("data.info:", data.info)
        map_data = map_data.reshape((data.info.width,data.info.height))
        print("af_reshape:", map_data.shape)
        #sl_Process.terminate()
        return map_data

    def get_costmap(self):

        print("waiting for costmap")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/move_base/local_costmap/costmap', OccupancyGrid, timeout=0.5)
            except:
                #print("NO MAP")
                #sl_Process.terminate()
                pass
        #print("data:", data)
        map_data = np.array(data.data)
        #print("bf_reshape:", map_data.shape)
        #print("data.info:", data.info)
        map_data = map_data.reshape((data.info.width,data.info.height))
        #print("af_reshape:", map_data.shape)

        print("obtained costmap")

        return map_data

    # Detect a collision from laser data
    def calculate_observation(self, data):
        min_range = 0.195
        min_laser = 2
        done = False
        col = False

        for i, item in enumerate(data.ranges):
            if min_laser > data.ranges[i]:
                min_laser = data.ranges[i]
            if (min_range > data.ranges[i] > 0):
                done = True
                col = True
        return done, col, min_laser

    def step(self, act):
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goalX
        marker.pose.position.y = self.goalY
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(0.1)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=0.5)
            except:
                pass

        dataOdom = None
        while dataOdom is None:
            try:
                dataOdom = rospy.wait_for_message('/odom', Odometry, timeout=0.5)
            except:
                pass


        local_costmap = self.get_costmap()

        rospy.wait_for_service('/gazebo/pause_physics')       
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)
        done, col, min_laser = self.calculate_observation(data)


        # Calculate robot heading from odometry data
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        quaternion = Quaternion(
            dataOdom.pose.pose.orientation.w,
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientation.y,
            dataOdom.pose.pose.orientation.z)
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)           # -pi, pi

        # Calculate distance to the goal from the robot
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        # Calculate the angle distance between the robots heading and heading toward the goal
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skewY < 0:
            beta = 0 - beta               # -pi, pi
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = -2*np.pi + beta2
        if beta2 < -np.pi:
            beta2 = 2*np.pi + beta2

        '''Bunch of different ways to generate the reward'''
        '''
        # reward = act[0]*0.7-abs(act[1])
        # r1 = 1 - 2 * math.sqrt(abs(beta2 / np.pi))
        # r2 = self.distOld - Dist
        r3 = lambda x: 1 - x if x < 1 else 0.0
        # rl = 0
        # for r in range(len(laser_state[0])):
        #    rl += r3(laser_state[0][r])
        # reward = 0.8 * r1 + 30 * r2 + act[0]/2 - abs(act[1])/2 - r3(min(laser_state[0]))/2
        reward = act[0] / 2 - abs(act[1]) / 2 - r3(min(laser_state[0])) / 2
        # reward = 30 * r2 + act[0] / 2 - abs(act[1]) / 2  # - r3(min(laser_state[0]))/2
        # reward = 0.8 * r1 + 30 * r2
        '''
        r6 = lambda x: 1 - x if x < 1 else 0.0
        r1 = 300*(self.distOld-Dist)
        r2 = 0
        r3 = 0
        r4 = -5
        r5 = act[0]/2 - abs(act[1]) / 2 - r6(min(laser_state[0])) / 2
        reward = r1 + r2 + r3 + r4 + r5

        self.distOld = Dist
        if Dist < 0.19:
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            r2 = 500
            reward = r1+r2+r3+r4+r5

        # Detect if ta collision has happened and give a large negative reward
        if col:
            r2 = -200
            reward = r1+r2+r3+r4+r5

        '''
        # Detect if the goal has been reached and give a large positive reward
        if Dist < 0.19:
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            reward = 80

        # Detect if ta collision has happened and give a large negative reward
        if col:
            reward = -100
        '''

        toGoal = [Dist, beta2, act[0], act[1], angle]
        state = np.append(laser_state, toGoal)
        #print(state)
        #print("(self.goalX, self.goalY)", self.goalX, self.goalY)
        #print("(self.odomX, self.odomY)", self.odomX, self.odomY)
        pocessed_costmap = local_costmap[8:72, 8:72] / 100
        return state, pocessed_costmap, reward, done, target


    def reset(self, sl_Process):

        sl_Process.terminate()
        time.sleep(1.0)

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
        self.random_box()
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        port = '11311'
        sl_Process_new = subprocess.Popen(["roslaunch", "-p", port, "/media/agent/eb0d0016-e15f-4a25-8c28-0ad31789f3cb/ROS/DRL-robot-navigation/TD3/assets/test_move_base5.launch"])
        time.sleep(4)

        data = None
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=0.5)
            except:
                pass
        
        local_costmap = self.get_costmap()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        # pre-process of scan data
        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)
        
        # pre-process for obtaining relative goal
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2
        toGoal = [Dist, beta2, 0.0, 0.0, angle]

        state = np.append(laser_state, toGoal)

        print("env reset done")
        processed_costmap = local_costmap[8:72, 8:72]/100

        processed_costmap[29:35, 30:34] = 100.0
        processed_costmap[30:34, 29:35] = 100.0

        return state, processed_costmap, sl_Process_new

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
        print("goal changed",self.goalX, self.goalY)

    # Randomly change the location of the boxes in the environment on each reset to randomize the training environment
    def random_box(self):
        for i in range(4):
            name = 'cardboard_box_' + str(i)

            x = 0
            y = 0
            chk = False
            while not chk:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                chk = check_pos(x, y)
                d1 = math.sqrt((x - self.odomX) ** 2 + (y - self.odomY) ** 2)
                d2 = math.sqrt((x - self.goalX) ** 2 + (y - self.goalY) ** 2)
                if d1 < 1.5 or d2 < 1.5:
                    chk = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def move(self):
        goal = MoveBaseGoal()
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
        sl_Process = subprocess.Popen(["roslaunch", "-p", port, "/media/agent/eb0d0016-e15f-4a25-8c28-0ad31789f3cb/ROS/DRL-robot-navigation/TD3/assets/test_move_base5.launch"])

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
            #state = self.move_base.get_state()
            #rospy.loginfo(state)
            rospy.loginfo("Goal succeeded!")
            sl_Process.terminate()
            return True
        '''
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        '''
        
