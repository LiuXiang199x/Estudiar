
import torch
import torch.nn as nn
import torch.nn.functional as F

import rospy

from typing import Any, Dict, List, Optional
import glob
import os
import time

import subprocess
from collections import deque
from test_env_exploration4 import GazeboEnv

import numpy as np
import gym.spaces as spaces
from base.spaces import EmptySpace, ActionSpace
from ext.rollout_storage import (
    RolloutStorageExtended,
)
import itertools
from base.rl.ppo import PPO


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
                .log_prob(actions.squeeze(-1))
                .view(actions.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def crop_map(h, x, crop_size, mode="bilinear"):
    """
    Crops a tensor h centered around location x with size crop_size

    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer

    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    """

    bs, _, H, W = h.size()
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
    start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
    end = start + crop_size - 1
    x_grid = (
        torch.arange(start, end + 1, step=1)
            .unsqueeze(0)
            .expand(crop_size, -1)
            .contiguous()
            .float()
    )
    y_grid = (
        torch.arange(start, end + 1, step=1)
            .unsqueeze(1)
            .expand(-1, crop_size)
            .contiguous()
            .float()
    )
    center_grid = torch.stack([x_grid, y_grid], dim=2).to(
        h.device
    )  # (crop_size, crop_size, 2)

    x_pos = x[:, 0] - Wby2  # (bs, )
    y_pos = x[:, 1] - Hby2  # (bs, )

    crop_grid = center_grid.unsqueeze(0).expand(
        bs, -1, -1, -1
    )  # (bs, crop_size, crop_size, 2)
    crop_grid = crop_grid.contiguous()

    # Convert the grid to (-1, 1) range
    crop_grid[:, :, :, 0] = (
                                    crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
                            ) / Wby2
    crop_grid[:, :, :, 1] = (
                                    crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
                            ) / Hby2

    h_cropped = F.grid_sample(h, crop_grid, mode=mode)

    return h_cropped


class GlobalPolicy(nn.Module):
    def __init__(self, G=240, use_data_parallel=False, gpu_ids=[]):
        super().__init__()

        self.G = G

        self.actor = nn.Sequential(  # (8, G, G)
            nn.Conv2d(8, 8, 3, padding=1),  # (8, G, G)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 5, padding=2),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, 5, padding=2),  # (2, G, G)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, padding=2),  # (1, G, G)
            Flatten(),  # (G*G, )
            # nn.Sigmoid(),  # frontier_mask
        )

        self.critic = nn.Sequential(  # (8, G, G)
            nn.Conv2d(8, 8, 3, padding=1),  # (8, G, G)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 5, padding=2),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, 5, padding=2),  # (2, G, G)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, padding=2),  # (1, G, G)
            Flatten(),
            nn.Linear(self.G * self.G, 1),
        )

        if use_data_parallel:
            self.actor = nn.DataParallel(
                self.actor, device_ids=gpu_ids, output_device=gpu_ids[0],
            )
            self.critic = nn.DataParallel(
                self.critic, device_ids=gpu_ids, output_device=gpu_ids[0],
            )

    def forward(self, inputs):
        raise NotImplementedError

    def _get_h12(self, inputs):
        x = inputs["pose_in_map_at_t"]
        h = inputs["map_at_t"]

        h_1 = crop_map(h, x[:, :2], self.G)
        h_2 = F.adaptive_max_pool2d(h, (self.G, self.G))

        h_12 = torch.cat([h_1, h_2], dim=1)

        return h_12

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        """
        Note: inputs['pose_in_map_at_t'] must obey the following conventions:
              origin at top-left, downward Y and rightward X in the map coordinate system.
        """
        M = inputs["map_at_t"].shape[2]
        h_12 = self._get_h12(inputs)
        '''
        action_logits = self.actor(h_12)
        dist = FixedCategorical(logits=action_logits)
        value = self.critic(h_12)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        '''
        action_logits = self.actor(h_12)
        dist = FixedCategorical(logits=action_logits)
        value = self.critic(h_12)
        action_probs = dist.probs
        free_mask = h_12[:, 4].reshape(h_12.shape[0], -1)
        action_probs3 = torch.where(free_mask == 0, action_probs, torch.ones_like(action_probs) * 0) #1e-7
        dist_3 = FixedCategorical(probs=action_probs3, validate_args=False)
        if deterministic:
            action = dist_3.mode()
        else:
            action = dist_3.sample()

        action_log_probs = dist_3.log_probs(action)

        '''  # frontier_mask
        action_logits = torch.clamp(self.actor(h_12), min=1e-4, max=1 - 1e-4)
        value = self.critic(h_12)
        action_mask = inputs["frontier_mask"]
        action_probs_mask = torch.where(action_mask == 1, action_logits, torch.ones_like(action_logits) * 1e-7)
        dist_2 = FixedCategorical(probs=action_probs_mask, validate_args=False)

        if deterministic:
            action = dist_2.mode()
        else:
            action = dist_2.sample()
        action_log_probs = dist_2.log_probs(action)
        ''' # frontier_mask
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, prev_actions, masks):
        h_12 = self._get_h12(inputs)
        value = self.critic(h_12)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, prev_actions, masks, action):
        h_12 = self._get_h12(inputs)
        '''
        action_logits = self.actor(h_12)
        dist = FixedCategorical(logits=action_logits)
        value = self.critic(h_12)

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
        '''

        '''  # frontier_mask
        action_logits = torch.clamp(self.actor(h_12), min=1e-4, max=1 - 1e-4)
        value = self.critic(h_12)
        action_mask = inputs["frontier_mask"]
        action_probs_mask = torch.where(action_mask == 1, action_logits, torch.ones_like(action_logits) * 1e-7)
        dist_2 = FixedCategorical(probs=action_probs_mask, validate_args=False)
        action_log_probs = dist_2.log_probs(action)

        dist_entropy = dist_2.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hxs
        '''  # frontier_mask

        action_logits = self.actor(h_12)
        dist = FixedCategorical(logits=action_logits)
        value = self.critic(h_12)
        action_probs = dist.probs
        free_mask = h_12[:, 0].reshape(h_12.shape[0], -1)
        action_probs3 = torch.where(free_mask == 0, action_probs, torch.ones_like(action_probs) * 0) #1e-7
        dist_3 = FixedCategorical(probs=action_probs3, validate_args=False)
        action_log_probs = dist_3.log_probs(action)
        dist_entropy = dist_3.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hxs


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

    # x_map = torch.clamp((Hby2 - y_world / map_scale), 0, H - 1).round()
    x_map = torch.clamp((Hby2 + y_world / map_scale), 0, H - 1).round()
    y_map = torch.clamp((Wby2 + x_world / map_scale), 0, W - 1).round()

    map_coors = torch.stack([x_map, y_map], dim=1)  # (bs, 2)

    return map_coors


def convert_map2world(map_coors, map_shape, map_scale):
    """
    This converts map coordinates to world coordinates
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_map = map_coors[0]
    y_map = map_coors[1]

    #y_world = (Hby2 - x_map) * map_scale
    #x_world = (y_map - Wby2) * map_scale

    y_world = (x_map - Hby2) * map_scale
    x_world = (y_map - Wby2) * map_scale

    world_coors = [x_world, y_world]
    return world_coors


def distance(x1, y1, x2, y2):
    """
    This calculates the distance between (x1, y1) and (x2, y2)
    """
    return ((x1 - y1) ** 2 + (x2 - y2) ** 2) ** 0.5


def measure_area_seen_performance(map_states, map_scale=1.0, reduction="mean"):
    """
    Inputs:
        map_states - (bs, 2, M, M) world map with channel 0 representing occupied
                     regions (1s) and channel 1 representing explored regions (1s)
    """

    bs = map_states.shape[0]
    explored_map = (map_states[:, 1] > 0.5).float()  # (bs, M, M)
    occ_space_map = (map_states[:, 0] > 0.5).float() * explored_map  # (bs, M, M)
    free_space_map = (map_states[:, 0] <= 0.5).float() * explored_map  # (bs, M, M)

    all_cells_seen = explored_map.view(bs, -1).sum(dim=1)  # (bs, )
    occ_cells_seen = occ_space_map.view(bs, -1).sum(dim=1)  # (bs, )
    free_cells_seen = free_space_map.view(bs, -1).sum(dim=1)  # (bs, )

    area_seen = all_cells_seen * (map_scale ** 2)
    free_space_seen = free_cells_seen * (map_scale ** 2)
    occupied_space_seen = occ_cells_seen * (map_scale ** 2)

    if reduction == "mean":
        area_seen = area_seen.mean().item()
        free_space_seen = free_space_seen.mean().item()
        occupied_space_seen = occupied_space_seen.mean().item()
    elif reduction == "sum":
        area_seen = area_seen.sum().item()
        free_space_seen = free_space_seen.sum().item()
        occupied_space_seen = occupied_space_seen.sum().item()

    return {
        "area_seen": area_seen,
        "free_space_seen": free_space_seen,
        "occupied_space_seen": occupied_space_seen,
    }


# =============== this function needs to be revised when observations need include more variables
def _create_global_rollouts(num_global_steps, num_envs, overall_map_size, map_size_for_g_policy):
    M = overall_map_size
    G = map_size_for_g_policy
    global_observation_space = spaces.Dict(
        {
            "pose_in_map_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(2,), dtype=np.float32
            ),
            "map_at_t": spaces.Box(
                low=0.0, high=1.0, shape=(4, M, M), dtype=np.float32
            ),
            # frontier_mask
            "frontier_mask": spaces.Box(
                low=0, high=2.0, shape=(G * G,), dtype=np.float32
            ),
        }
    )
    global_action_space = ActionSpace(
        {
            f"({x[0]}, {x[1]})": EmptySpace()
            for x in itertools.product(range(G), range(G))
        }
    )
    global_rollouts = RolloutStorageExtended(
        num_global_steps,
        num_envs,
        global_observation_space,
        global_action_space,
        1,
        enable_recurrence=False,
        delay_observations_entry=True,
        delay_masks_entry=True,
        enable_memory_efficient_mode=True,
    )
    return global_rollouts


def _process_maps_frontier(maps, G):  # maps: global_map, (G,G): policy output's size   # frontier_mask
    maps = F.adaptive_max_pool2d(maps, (G, G))
    '''
    thresh_obstacle = 0.6
    thresh_explored = 0.6
    free_mask = (maps[:, 0] <= thresh_obstacle) & (
            maps[:, 1] > thresh_explored
    )
    '''
    bs = maps.shape[0]
    action_mask = maps.view(bs, -1)  # returns free_mask, this needs to be changed if frontier_mask is to be returned
    return action_mask


def _create_global_policy_inputs(global_map, visited_states, map_xy, frontier_map):
    """
    global_map     - (bs, 2, V, V) - map occupancy, explored states
    visited_states - (bs, 1, V, V) - agent visitation status on the map
    map_xy   - (bs, 2) - agent's XY position on the map
    """
    agent_map_x = map_xy[:, 0].long()  # (bs, )
    agent_map_y = map_xy[:, 1].long()  # (bs, )
    agent_position_onehot = torch.zeros_like(visited_states)
    agent_position_onehot[:, 0, agent_map_y, agent_map_x] = 1
    h_t = torch.cat(
        [global_map, visited_states, agent_position_onehot], dim=1
    )  # (bs, 4, M, M)

    frontier_mask = _process_maps_frontier(frontier_map, 240)  # frontier_mask
    global_policy_inputs = {
        "pose_in_map_at_t": map_xy,
        "map_at_t": h_t,
        "frontier_mask": frontier_mask,  # frontier_mask
    }

    return global_policy_inputs


def load_checkpoint(checkpoint_path: str, *args, **kwargs) -> Dict:
    r"""Load checkpoint of specified path as a dict.

    Args:
        checkpoint_path: path of target checkpoint
        *args: additional positional args
        **kwargs: additional keyword args

    Returns:
        dict containing checkpoint info
    """
    return torch.load(checkpoint_path, *args, **kwargs)


def measure_area_seen_performance(map_states, map_scale=1.0, reduction="mean"):
    """
    Inputs:
        map_states - (bs, 2, M, M) world map with channel 0 representing occupied
                     regions (1s) and channel 1 representing explored regions (1s)
    """

    bs = map_states.shape[0]
    explored_map = (map_states[:, 1] > 0.5).float()  # (bs, M, M)
    occ_space_map = (map_states[:, 0] > 0.5).float() * explored_map  # (bs, M, M)
    free_space_map = (map_states[:, 0] <= 0.5).float() * explored_map  # (bs, M, M)

    all_cells_seen = explored_map.view(bs, -1).sum(dim=1)  # (bs, )
    occ_cells_seen = occ_space_map.view(bs, -1).sum(dim=1)  # (bs, )
    free_cells_seen = free_space_map.view(bs, -1).sum(dim=1)  # (bs, )

    area_seen = all_cells_seen * (map_scale ** 2)
    free_space_seen = free_cells_seen * (map_scale ** 2)
    occupied_space_seen = occ_cells_seen * (map_scale ** 2)

    if reduction == "mean":
        area_seen = area_seen.mean().item()
        free_space_seen = free_space_seen.mean().item()
        occupied_space_seen = occupied_space_seen.mean().item()
    elif reduction == "sum":
        area_seen = area_seen.sum().item()
        free_space_seen = free_space_seen.sum().item()
        occupied_space_seen = occupied_space_seen.sum().item()

    return {
        "area_seen": area_seen,
        "free_space_seen": free_space_seen,
        "occupied_space_seen": occupied_space_seen,
    }


if __name__ == '__main__':
    torch.manual_seed(2.0)
    #torch.backends.cudnn.benchmark = False
    # Basic config
    NUM_GLOBAL_UPDATES = 1000  # number of episodes
    NUM_GLOBAL_STEPS = 20 #10
    overall_map_scale = 0.05
    overall_map_size = 1201  # 481 # original global map size = 2*M*M
    map_size_for_g_policy = 240
    G = map_size_for_g_policy
    num_envs = 1
    h_rgb = 128
    w_rgb = 128
    h_depth = 128
    w_depth = 128
    MAX_GLOBAL_STEPS = 20 #20 #20

    # config parameters for PPO
    clip_param = 0.2
    ppo_epoch = 1
    num_mini_batch = 1
    value_loss_coef = 0.5
    entropy_coef = 0.001
    lr = 0.00025
    eps = 0.00001
    max_grad_norm = 0.5
    use_gae = True
    gamma = 0.99
    tau = 0.95
    reward_window_size = 50
    loss_stas_window_size = 100
    global_reward_scale = 0.0001

    # save checkpoint
    CHECKPOINT_FOLDER = os.path.join(os.path.dirname(__file__), "new_checkpoints")
    CHECKPOINT_INTERVAL = 1
    # count_checkpoints = 0

    # devices config
    device1 = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    #device1 = torch.device("cpu")
    #global_policy_gpu_ids = [0]
    #device = global_policy_gpu_ids[0]

    # ===============create env==================
    # Set the parameters for the implementation
    env_name = "HalfCheetahBulletEnv-v0"  # Name of the PyBullet environment. The network is updated for HalfCheetahBulletEnv-v0
    # Create the training environment
    env = GazeboEnv('test_move_base8.launch', 1, 1, 1)
    time.sleep(4)
    port = '11311'


    # create global states, for caching the online states of global agent, which could be used for computing rewards
    M = overall_map_size
    s = overall_map_scale
    ground_truth_states = {
        # To measure area seen
        "visible_occupancy": torch.zeros(
            num_envs, 2, M, M, device=device1
        ),
        "pose": torch.zeros(num_envs, 3, device=device1),
        "prev_area_seen": torch.zeros(
            num_envs, 1, device=device1
        ),
        "pre_total_path": torch.zeros(
            num_envs, 1, device=device1
        ),
    }

    global_policy = GlobalPolicy(G=map_size_for_g_policy, use_data_parallel=False, gpu_ids=[])
    global_policy.to(device1)
    global_agent = PPO(
        actor_critic=global_policy,
        clip_param=clip_param,
        ppo_epoch=ppo_epoch,
        num_mini_batch=num_mini_batch,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        lr=lr,
        eps=eps,
        max_grad_norm=max_grad_norm,
    )
    # create global rollouts, for caching the online training data for global agent
    global_rollouts = _create_global_rollouts(NUM_GLOBAL_STEPS, num_envs, overall_map_size, map_size_for_g_policy)
    global_rollouts.to(device1)

    masks = torch.zeros(num_envs, 1, device=device1)

    current_global_episode_reward = torch.zeros(num_envs, 1)

    # num_updates_start = 0
    # update = 0

    t_start = time.time()
    env_time = 0
    pth_time = 0
    # count_steps = 0

    # ==================== Measuring memory consumption ===================
    total_memory_size = 0
    print("================== Global policy rollouts ====================")
    for k, v in global_rollouts.observations.items():
        mem = v.element_size() * v.nelement() * 1e-9
        print(f"key: {k:<40s}, memory: {mem:>10.4f} GB")
        total_memory_size += mem
    print(f"Total memory: {total_memory_size:>10.4f} GB")

    # Resume checkpoint if available
    checkpoints = glob.glob(f"{CHECKPOINT_FOLDER}/*.pth")
    if len(checkpoints) == 0:
        count_steps = 0
        count_checkpoints = 0
        update = 0
        num_episodes_start = 0
    else:
        last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
        checkpoint_path = last_ckpt
        # Restore checkpoints to models
        ckpt_dict = load_checkpoint(checkpoint_path)
        global_agent.load_state_dict(ckpt_dict["global_state_dict"])
        global_policy = global_agent.actor_critic
        # Set the logging counts
        ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
        update = ckpt_dict["extra_state"]["update"] + 1
        num_episodes_start = ckpt_dict["extra_state"]["num_episode"] + 1
        count_steps = ckpt_dict["extra_state"]["step"]
        count_checkpoints = ckpt_id + 1
        print("num_episodes_start:", num_episodes_start)
        print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

    sl_Process = None

    for ep_num in range(num_episodes_start, NUM_GLOBAL_UPDATES):

        observations, sl_Process = env.reset(sl_Process)
        time.sleep(4)

        dones = torch.zeros(num_envs, 1)
        rospy.loginfo("Observations are loaded")

        for step in range(MAX_GLOBAL_STEPS):
            print("--1--:Epoch:{}, step:{}".format(ep_num, step))

            # call observations for global policy
            global_policy.eval()
            with torch.no_grad():
                # ================================ act at step
                global_map = observations["global_map"].to(device1)  # (bs, 2, M, M)
                global_pose = observations["global_pose"].to(device1)  # (bs, 3)
                cur_collison_map = observations["collision_map"].to(device1)  # (bs, M, M)
                cur_visited_map = observations["visited_map"].to(device1)  # (bs, 1, M, M),  a path in single-cell size

                frontier_map = observations["frontier_map"].to(device1)  # (bs, M, M)
                M = global_map.shape[2]
                # process visited map with agent's radius, pro_visited_map will represent all cells visited by the agent with a radius
                # pro_visited_map = torch.zeros_like(cur_collison_map)  # (bs, M, M),
                #for i in range(num_envs):
                #    for j in range(M):
                #        for k in range(M):
                #            if cur_visited_map[i, 0, j, k] == 1:
                #                pro_visited_map[i, j - 3:j + 4, k - 3:k + 4] = 1

                map_xy = convert_world2map(global_pose[:, :2], (M, M), s)
                map_xy = torch.clamp(map_xy, 0, M - 1)
                curr_map_position = map_xy

                ####### process global_map before being delivered to global policy, enhancement from me
                # global_map[:, 0, :, :][cur_collison_map == 1] = 1
                # global_map[:, 1, :, :][cur_collison_map == 1] = 1
                # global_map[:, 0, :, :][pro_visited_map == 1] = 0
                # global_map[:, 1, :, :][pro_visited_map == 1] = 1

                # ====================== Global policy action selection =======================

                global_policy_inputs = _create_global_policy_inputs(
                    global_map, cur_visited_map, curr_map_position, frontier_map
                )
                (
                    global_value,
                    global_action,
                    global_action_log_probs,
                    _,
                ) = global_policy.act(global_policy_inputs, None, None, None)
                # Convert action to location (row-major format)

                global_action_map_x = (global_action.squeeze(1) % G).float()  # (bs, )
                global_action_map_y = (global_action.squeeze(1) / G).float()  # (bs, )
                # Convert to MxM map coordinates
                global_action_map_x = global_action_map_x * M / G
                global_action_map_y = global_action_map_y * M / G
                global_action_map_xy = torch.stack(
                    [global_action_map_x, global_action_map_y], dim=1
                )

                global_policy_outputs = {
                    "values": global_value,
                    "actions": global_action,
                    "action_log_probs": global_action_log_probs,
                }

            # Convert action to goal
            rospy.loginfo("Convert action to goal")

            global_action_map_xy = torch.clamp(global_action_map_xy, 0, M - 1).round()

            goal_x = int(global_action_map_xy[0][0].item())
            goal_y = int(global_action_map_xy[0][1].item())
            #goal = [goal_x, goal_y]

            goal = [goal_y, goal_x]
            explored_map = (global_map[:, 1] > 0.5).float()  # (bs, M, M)
            free_space_map = (global_map[:, 0] <= 0.5).float() * explored_map  # (bs, M, M)

            if free_space_map[0, goal_y, goal_x] != 1 or free_space_map[0, goal_y, goal_x] != 0: #if free_space_map[0, goal_y, goal_x] != 1:
                for i in [0, 1, -1, 2, -2, 3, -3]:
                    for j in [0, 1, -1, 2, -2, 3, -3]:
                        if 0 <= goal_y+i < M and 0 <= goal_x+j < M:
                            if free_space_map[0, goal_y+i, goal_x+j] == 1:
                                if free_space_map[0, goal_y, goal_x] != 1:
                                    goal = [goal_y+i, goal_x+j]
                                if torch.sum(free_space_map[0, goal_y+i-2:goal_y+i+3, goal_x+j-2:goal_x+j+3], dim=(0, 1)).item() >= 22:
                                    goal = [goal_y+i, goal_x+j]
                                    print("=======Goal migrated==========")
                                    break
                                #break
                    else:
                        continue
                    break


            rospy.loginfo("Selected goal in map:" + str(goal_x) + str(' ') + str(goal_y))

            goal = convert_map2world(goal, (M, M), s)
            goal_x = goal[0]
            goal_y = goal[1]
            rospy.loginfo("Selected goal:" + str(goal_x) + str(' ') + str(goal_y))

            # Update states and compute rewards for last step
            ground_truth_states["visible_occupancy"].copy_(observations["global_map"])
            ground_truth_states["pose"].copy_(observations["global_pose"])
            cur_area_seen = measure_area_seen_performance(
                ground_truth_states["visible_occupancy"], reduction="none"
            )["area_seen"]

            cur_total_path = torch.sum(cur_visited_map, dim=(1, 2, 3)) #* overall_map_scale
            #global_path_length = observations["path_length"] * overall_map_scale
            global_path_length = observations["path_length"]

            if global_rollouts.step == 0:
                global_rewards = torch.zeros(num_envs, 1)
                #global_path_length = torch.zeros(num_envs, 1)
            else:
                global_rewards = (
                        cur_area_seen - ground_truth_states["prev_area_seen"]
                ).cpu()
                #global_path_length = (cur_total_path - ground_truth_states["pre_total_path"]).cpu()

            current_global_episode_reward += global_rewards

            global_rewards -= (global_path_length * 10).cpu()  # path_reward
            # global_rewards += current_global_episode_reward / 4  # path_reward   ???
            global_rewards += current_global_episode_reward.sqrt() * 2 * 10

            ground_truth_states["prev_area_seen"].copy_(
                cur_area_seen
            )
            ground_truth_states["pre_total_path"].copy_(cur_total_path)

            global_rollouts.rewards[global_rollouts.step - 1].copy_(
                global_rewards * global_reward_scale
            )
            global_rollouts.insert(
                global_policy_inputs,
                None,
                global_policy_outputs["actions"],
                global_policy_outputs["action_log_probs"],
                global_policy_outputs["values"],
                torch.zeros_like(global_rewards),
                masks.to(device1),
            )

            #current_global_episode_reward += global_rewards

            observations, sl_Process = env.step(goal, sl_Process)

            dones = torch.sum(frontier_map, dim=(1, 2)) == 0
            if dones[0]:
                print("==========Finish searching global area==========")

            masks.copy_(
                torch.tensor(
                    [[0.0] if done else [1.0] for done in dones], dtype=torch.float
                )
            )

            print("--5--:Epoch:{}, step:{}".format(ep_num, step))
            if step == MAX_GLOBAL_STEPS - 1:
                masks.fill_(0)

            # env.pub_result()

            # ======================= update global policy =============================================
            if step % NUM_GLOBAL_STEPS == NUM_GLOBAL_STEPS - 1:
                try:
                    assert global_rollouts.step == NUM_GLOBAL_STEPS
                except:
                    rospy.loginfo("======AssertionError======")
                global_policy.eval()
                with torch.no_grad():
                    global_policy_inputs = _create_global_policy_inputs(
                        global_map, cur_visited_map, curr_map_position, frontier_map
                    )

                global_policy.train()
                # assert episode_step_count[0].item() % goal_interval == 0
                assert global_policy_inputs is not None
                for k, v in global_policy_inputs.items():
                    global_rollouts.observations[k][global_rollouts.step].copy_(v)

                cur_area_seen = measure_area_seen_performance(
                    ground_truth_states["visible_occupancy"], reduction="none"
                )["area_seen"]

                global_rewards = (
                        cur_area_seen - ground_truth_states["prev_area_seen"]
                ).cpu()
                global_rollouts.rewards[global_rollouts.step - 1].copy_(
                    global_rewards * global_reward_scale
                )
                global_rollouts.masks[global_rollouts.step].copy_(masks)

                # assert global_rollouts.step == NUM_GLOBAL_STEPS
                t_update_model = time.time()
                with torch.no_grad():
                    last_observation = {
                        k: v[-1].to(device1)
                        for k, v in global_rollouts.observations.items()
                    }
                    next_global_value = global_policy.get_value(
                        last_observation,
                        None,
                        global_rollouts.prev_actions[-1].to(device1),
                        global_rollouts.masks[-1].to(device1),
                    ).detach()

                global_rollouts.compute_returns(
                    next_global_value, use_gae, gamma, tau
                )

                (
                    global_value_loss,
                    global_action_loss,
                    global_dist_entropy,
                ) = global_agent.update(global_rollouts)
                update += 1
                update_metrics = {
                    "value_loss": global_value_loss,
                    "action_loss": global_action_loss,
                    "dist_entropy": global_dist_entropy,
                }
                print('update %d ' % update)
                print('value_loss:', update_metrics["value_loss"])
                print('action_loss:', update_metrics["action_loss"])
                print('dist_entropy:', update_metrics["dist_entropy"])

                global_rollouts.after_update()

                delta_pth_time = time.time() - t_update_model

                if update % CHECKPOINT_INTERVAL == 0:
                    checkpoint = {
                        "global_state_dict": global_agent.state_dict(),
                        "extra_state": dict(step=count_steps, update=update, num_episode=ep_num),
                    }
                    torch.save(checkpoint, os.path.join(CHECKPOINT_FOLDER, f"ckpt.{count_checkpoints}.pth"))
                    count_checkpoints += 1
                    print("SAVE THE MODELLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")

            if step == MAX_GLOBAL_STEPS - 1:
                dones = torch.zeros(num_envs, 1)
                rospy.loginfo("Observations are loaded")

                for k in ground_truth_states.keys():
                    ground_truth_states[k].fill_(0)
                # ground_truth_states["visible_occupancy"].copy_(observations["global_map"])
                # ground_truth_states["pose"].copy_(observations["global_pose"])
                #count_steps += num_envs
                #break
            count_steps += num_envs

    env.gz_Process.terminate()
    #sl_Process.terminate()
