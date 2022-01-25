import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from collections import deque
from test_env_map import GazeboEnv

# Set the parameters for the implementation
env_name = "HalfCheetahBulletEnv-v0"  # Name of the PyBullet environment. The network is updated for HalfCheetahBulletEnv-v0

# Create the training environment
env = GazeboEnv('test_move_base4.launch', 1, 1, 1)
time.sleep(10)

port = '11311'
sl_Process = subprocess.Popen(["roslaunch", "-p", port, "/media/agent/eb0d0016-e15f-4a25-8c28-0ad31789f3cb/ROS/DRL-robot-navigation/TD3/assets/test_move_base5.launch"])
time.sleep(4)

_, _, sl_Process = env.reset(sl_Process)
for i in range(500):
    state, local_costmap, reward, done, target = env.step([0.1, 0.1])



sl_Process.terminate()

print("state:", state)
print("costmap", local_costmap)
print("costmap.shape", local_costmap.shape)
print(reward)
print(done)
print(target)



'''
eval_episodes = 400  # 400 # number of episodes for evaluation
avg_reward = 0.
col = 0
for _ in range(eval_episodes):
    time.sleep(2)
    count = 0
    state = env.reset()
    done = False
    goal_reached = env.move()
    count += 1
    if goal_reached:
        col += 0
    else:
        col += 1
avg_col = col/eval_episodes
print("..............................................")
print("Average Reward over %i Evaluation Episodes: %f" % (eval_episodes, avg_col))
print("..............................................")

'''
