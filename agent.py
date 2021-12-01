import abc
import utils 
import environment

from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import random
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, id, env_params, spt=None):
        self.env_params = env_params
        self.spt = spt
        self.location = (0,0)
        self.prev_location = (0,0)
        self.food = 0
        self.active = 0
        self.id = id
        self.bfs_active = 0
    
    def get_state(self):
        return self.location, self.food

    def set_spt(self, spt):
        self.spt = spt
    
    @abc.abstractmethod
    def get_action(self, observation, valid_movements):
        # Returns movement, pheromone
        pass 

    def __str__(self):
        return ''+ str(self.location) + ' ' + str(self.active) + ' ' + str(self.food)

class RandomAgent(Agent):
    def __init__(self, id, env_params, spt=None):
        super().__init__(id, env_params, spt)
        self.orientation = None
        self.obs_rad = self.env_params['observation_radius']
        self.obs_window = self.env_params['observation_radius']*2+1
    def get_action(self, observation, valid_movements, fwd_only = 1):
        return random.choice(valid_movements), random.random()
    
    
class SwarmAgent(Agent):
    def __init__(self, id, env_params, spt=None):
        super().__init__(id, env_params, spt)
        self.orientation = None
        self.obs_rad = self.env_params['observation_radius']
        self.obs_window = self.env_params['observation_radius']*2+1
        
    def flatten(self, obs_r, obs_c):
        return self.obs_window * r + c
    
    def unflatten(self, obs_pos):
        return (obs_pos // self.obs_window, obs_pos % self.obs_window)
        
    def get_action(self, observation, valid_movements, fwd_only = 1):
        agent_grid, static_grid, dynamic_grid = observation
        d = dynamic_grid.copy()
        last_action = tuple(np.array(self.location) - np.array(self.prev_location))
        if self.food == 1 and self.location != self.env_params['coding_dict']['hive']:
            # choose direction with BFS
            lay_pheromone = self.env_params['pheromone']['step_if_food']
            good_actions = self.spt[self.location[0]][self.location[1]]
            next_movement = good_actions[np.random.randint(len(good_actions))]
        else:
            lay_pheromone = self.env_params['pheromone']['step']
            # move forward, probs weighted by pheromone strength
            if last_action == (-1,0):
                d[self.obs_rad + 1 - fwd_only:, :] = 0

            elif last_action == (0,1):
                d[:, :self.obs_rad + fwd_only] = 0

            elif last_action == (1,0):
                d[:self.obs_rad + fwd_only, :] = 0

            elif last_action == (0,-1):
                d[:, self.obs_rad + 1 - fwd_only:] = 0
        
            dflat = d.flatten()
            if np.sum(dflat) == 0:
                probs = [1/len(dflat)] * len(dflat)
            else:
                probs = dflat/np.sum(dflat)

            # choose random target location in fwd direction, weighted by pheromone
            best_loc = np.random.choice(np.arange(len(dflat)), p = probs)
            best_loc = self.unflatten(best_loc)
                
            good_actions = []     
            # if best action row > agent row in observation window
            if best_loc[0] > self.obs_rad:
                good_actions.append((1,0))
            if best_loc[0] < self.obs_rad:
                good_actions.append((-1,0))
            if best_loc[1] > self.obs_rad:
                good_actions.append((0,1))
            if best_loc[1] < self.obs_rad:
                good_actions.append((0,-1))

            assert(len(valid_movements) > 0)
            # if list empty take random action from all actions not blocked by obstacles
            good_actions = list(set(good_actions).intersection(set(valid_movements)))
            if len(good_actions) > 0:
                next_movement = good_actions[np.random.randint(len(good_actions))]
            else:
                next_movement = valid_movements[np.random.randint(len(valid_movements))]

        return next_movement, lay_pheromone

# DQN based on tutorial from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(DEVICE)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

class DQNAgent(Agent):
    def __init__(self, id, env_params, net_filepath=None, spt=None):
        super().__init__(id, env_params, spt)

        obs_window = self.env_params['observation_radius']*2+1
        
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        self.screen_height, self.screen_width = 100, 200

        self.environment_actions = self.env_params['env_actions']
        self.n_actions = len(self.environment_actions)
        self.policy_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(DEVICE)
        self.target_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(DEVICE)
        
        if net_filepath is not None:
            self.policy_net.load_state_dict(torch.load(net_filepath))

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())   
        self.steps_done = 0
    
    def get_action(self, observation, valid_movements, train=False):
        assert(len(valid_movements) > 0)
        if self.food == 1 and self.location != self.env_params['coding_dict']['hive']:
            # choose direction with BFS
            lay_pheromone = self.env_params['pheromone']['step_if_food']
            good_actions = self.spt[self.location[0]][self.location[1]]
            next_movement = good_actions[np.random.randint(len(good_actions))]

            if train:
                return torch.tensor([[self.environment_actions.index(next_movement)]], device=DEVICE, dtype=torch.long), lay_pheromone
            else:
                return next_movement, lay_pheromone
            
        elif self.food == 0:
            lay_pheromone = self.env_params['pheromone']['step']

            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            if (sample > eps_threshold and train) or (not train):
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    sorted_tensor, opt_action_ids = torch.sort(self.policy_net(observation)[0], descending=True)
                    valid_movement_ids = [self.environment_actions.index(action) for action in valid_movements]
                    opt_action_id = None
                    for i in range(self.n_actions):
                        action = opt_action_ids[i].item()
                        if action in valid_movement_ids:
                            opt_action_id = opt_action_ids[i]
                            break
                    #print(list(opt_action_ids), valid_movement_ids, opt_action_id)
                    if train:
                        return opt_action_id.view(1,1), lay_pheromone
                    else:
                        return self.environment_actions[int(opt_action_id)], lay_pheromone
            else:
                if train:
                    return torch.tensor([[self.environment_actions.index(random.choice(valid_movements))]], device=DEVICE, dtype=torch.long), lay_pheromone
                else:
                    return random.choice(valid_movements), lay_pheromone


    def optimize_model(self, shared_memory):
        if len(shared_memory) < self.BATCH_SIZE:
            return
        
        transitions = shared_memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


        


