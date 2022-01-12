import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

#from velodyne_env import GazeboEnv
from velodyne_env_new import GazeboEnv

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.max_action = max_action
        self.soft = nn.Softsign()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
env_name = "HalfCheetahBulletEnv-v0"  # Name of the PyBullet environment. The network is updated for HalfCheetahBulletEnv-v0
seed = 0  # Random seed number


# Create the training environment
env = GazeboEnv('multi_robot_scenario.launch', 1, 1, 1)
time.sleep(5)
# env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = 24
action_dim = 2
max_action = 1

# Create and load the actor network
file_name = "TD3_velodyne"  # name of the file to store the policy
actor = Actor(state_dim, action_dim, max_action).to(device)
actor.load_state_dict(torch.load('%s/%s_actor.pth' % ("./results", file_name)))

max_ep = 500
eval_episodes = 4 # 1  # number of episodes for evaluation
avg_reward = 0.
col = 0
for _ in range(eval_episodes):
    count = 0
    state = env.reset()
    done = False
    while not done and count < max_ep+1:
        state_tensor = torch.Tensor(np.array(state).reshape(1, -1)).to(device)
        #print(state_tensor)
        #print(state_tensor.shape)
        action = actor(state_tensor).cpu().data.numpy().flatten()
        a_in = [(action[0] + 1) / 2, action[1]]
        state, reward, done, _ = env.step(a_in)
        avg_reward += reward
        count += 1
        if reward < -90:
            col += 1
avg_reward /= eval_episodes
avg_col = col/eval_episodes
print("..............................................")
print("Average Reward over %i Evaluation Episodes: %f, %f" % (eval_episodes, avg_reward, avg_col))
print("..............................................")
