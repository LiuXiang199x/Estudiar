import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from collections import deque
from test_env import GazeboEnv

# Set the parameters for the implementation
env_name = "HalfCheetahBulletEnv-v0"  # Name of the PyBullet environment. The network is updated for HalfCheetahBulletEnv-v0

# Create the training environment
env = GazeboEnv('test_move_base4.launch', 1, 1, 1)
time.sleep(10)


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


