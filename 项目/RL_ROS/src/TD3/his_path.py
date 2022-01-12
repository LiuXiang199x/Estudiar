#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry



def convert_world2map(world_coors, map_shape, map_scale):
    """
    World coordinate system:
        Agent starts at (0, 0) facing upward along X. Y is rightward.
    Map coordinate system:
        Agent starts at (W/2, H/2) with X rightward and Y downward.

    Inputs:
        world_coors: (bs, 2) --- (x, y) in world coordinates
        map_shape: tuple with (H, W)
        map_scale: scalar indicating the cell size in the map
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_world = world_coors[:, 0]
    y_world = world_coors[:, 1]

    # x_map = torch.clamp((Wby2 + y_world / map_scale), 0, W - 1).round()
    # y_map = torch.clamp((Hby2 - x_world / map_scale), 0, H - 1).round()
    x_map = np.clip((Hby2 - y_world / map_scale), 0, H - 1).round()
    y_map = np.clip((Wby2 + x_world / map_scale), 0, W - 1).round()

    map_coors = np.stack([x_map, y_map], dim=1)  # (bs, 2)

    return map_coors


def callback(data, prev_path_map):
    path_pub = rospy.Publisher('path_map', OccupancyGrid, queue_size=1)
    odomX = data.pose.pose.position.x
    odomY = data.pose.pose.position.y
    prev_map = pre_path_map

