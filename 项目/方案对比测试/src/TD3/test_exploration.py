import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from collections import deque
from test_env_exploration4 import GazeboEnv

# Set the parameters for the implementation
env_name = "HalfCheetahBulletEnv-v0"  # Name of the PyBullet environment. The network is updated for HalfCheetahBulletEnv-v0

# Create the training environment
env = GazeboEnv('test_move_base8.launch', 1, 1, 1)
time.sleep(4)
port = '11311'
sl_Process = None

for i in range(55):
    print("======== Number of episode ========= ", i + 1)
    state, sl_Process = env.reset(sl_Process)
    state, sl_Process = env.step([0.0-1, 0.0-1], sl_Process)
    print("======== Number of episode ========= ", i+1)

env.gz_Process.terminate()


