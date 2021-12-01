import numpy as np
from queue import Queue
import random
import copy

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def calculate_food(grid, env_params):
    counts = np.bincount(grid.flatten().astype('int64'))
    food_counts = counts[env_params['coding_dict']['food_start']:]
    assert(len(food_counts) <= env_params['max_food'])
    if len(food_counts) != env_params['max_food']:
        food_counts = np.append(food_counts, np.zeros(env_params['max_food']-len(food_counts)))
    total_food = np.dot(np.arange(1, env_params['max_food']+1), food_counts)
    return total_food

def calculate_blockades(grid, env_params):
    counts = np.bincount(grid.flatten().astype('int64'))
    return counts[env_params['coding_dict']['blockade']]

def check_valid(grid, env_params, throw_error=True):
    assert(grid[0][0] == env_params['coding_dict']['hive'])
    assert(grid[1][0] == env_params['coding_dict']['hive'])
    assert(grid[0][1] == env_params['coding_dict']['hive'])
    counts = np.bincount(grid.flatten().astype('int64'))
    num_blockades = calculate_blockades(grid, env_params)
    total_food = calculate_food(grid, env_params)
    if throw_error:
        assert(num_blockades <= env_params['grid']['blockade'])
        assert(total_food == env_params['grid']['food'])
    else:
        if ((num_blockades <= env_params['grid']['blockade']) 
                and (total_food == env_params['grid']['food'])):
            return True
        return False
    
def get_food(cell_value, env_params):
    # helper method transforming cell value to amount of food
    assert(cell_value >= env_params['coding_dict']['food_start'])
    food = cell_value-env_params['coding_dict']['food_start']+1
    assert(food <= env_params['max_food'])
    return food

def get_baseline_food(cell_value, total_food, env_params):
    # helper method to calculate total food minus the current cell's food
    total_food = total_food-get_food(cell_value, env_params)

def get_grid_food(food, env_params):
    # helper method transforming amount of food to cell value
    assert(food <= env_params['max_food'])
    return env_params['coding_dict']['food_start']+food-1

def generate_random_grid(env_params):
    # randomly select the number of blockades
    num_blockades = random.randint(1, env_params['grid']['blockade'])

    grid = np.zeros(env_params['N']*env_params['M']-num_blockades)

    # insert the blockades into the grid
    grid = np.insert(grid, np.random.choice(len(grid), size=num_blockades), np.ones(num_blockades))
    grid = grid*env_params['coding_dict']['blockade']

    # place hive at (0,0), (1,0), and (0,1)
    grid[0] = env_params['coding_dict']['hive']
    grid[1] = env_params['coding_dict']['hive']
    grid[env_params['M']] = env_params['coding_dict']['hive']

    # place food across the grid
    food_left = env_params['grid']['food']
    potential_food_idxs = list(np.where(grid == env_params['coding_dict']['empty'])[0])
    random.shuffle(potential_food_idxs)
    while food_left > 0:
        food = random.randint(1, min(food_left, env_params['max_food']))
        food_left -= food
        grid[potential_food_idxs[0]] = get_grid_food(food, env_params)
        del potential_food_idxs[0]

    grid = np.reshape(grid, (env_params['N'],env_params['M']))

    check_valid(grid, env_params)
    return grid

def process_grids(observation, env_params, visual=True):
    agent_grid, static_grid, dynamic_grid = observation
    N = env_params['N']
    M = env_params['M']
    if visual:
        agent_static_grid_processed, dynamic_grid_processed = np.zeros((N, M, 3)), np.zeros((N, M, 3))
    else:
        agent_static_grid_processed, dynamic_grid_processed = np.zeros((3, N, M)), np.zeros((3, N, M))
    
    def set_channel(a, b, grid, color):
        for channel in range(3):
            grid[channel][a][b] = color[channel]
    
    for i in range(N):
        for j in range(M):
            # combine agent and static grids
            if static_grid[i,j] == env_params['coding_dict']['empty'] and agent_grid[i,j] == env_params['coding_dict']['agent']:
                if visual:
                    agent_static_grid_processed[i,j] = env_params['rgb_coding'][env_params['coding_dict']['agent']]
                else:
                    set_channel(i, j, agent_static_grid_processed, env_params['rgb_coding'][env_params['coding_dict']['agent']])
            else:
                if visual:
                    agent_static_grid_processed[i,j] = env_params['rgb_coding'][static_grid[i,j]]
                else:
                    set_channel(i, j, agent_static_grid_processed, env_params['rgb_coding'][static_grid[i,j]])
            # compute greyscale values for dynamic grid
            greyscale_val = 255 * (env_params['pheromone']['cap'] - dynamic_grid[i,j])/env_params['pheromone']['cap']
            assert(greyscale_val > 0 and greyscale_val <= 255)
            if visual:
                dynamic_grid_processed[i][j] = [greyscale_val]*3
            else:
                set_channel(i, j, dynamic_grid_processed, [greyscale_val]*3)
    return (agent_static_grid_processed, dynamic_grid_processed)

def prepare_observation(observation, env_params, resize_shape=None, training=True):
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(resize_shape, interpolation=Image.CUBIC),
                    T.ToTensor()])
    
    # Add color channels
    agent_static_grid, dynamic_grid = process_grids(observation, env_params, visual=False)
    appended_grid = np.append(agent_static_grid, dynamic_grid, axis=0)
    # Change H,W,C --> C,H,W
    appended_grid = appended_grid.transpose((2, 0, 1))
    appended_grid = torch.from_numpy(appended_grid)
    
    if training:
        # Add a batch dimension --> B, C, H, W
        return resize(appended_grid).unsqueeze(0)
    else:
        return resize(appended_grid)

# get array of movements to take from each square towards hive
# avoids obstacles (coded as 4 by default)
# desc: static grid; loc: use (0,0) to go to hive
def expert_navigation_policy_set(desc, loc, env_params):
    obstacle_code = env_params['coding_dict']['blockade']
    (loc_r, loc_c) = loc
    num_rows = len(desc)
    num_cols = len(desc[0])
    max_row = num_rows - 1
    max_col = num_cols - 1
    spt = [[[[], np.inf] for _ in range(num_cols)] for _ in range(num_rows)]
    
    
#     spt[loc_r][loc_c][0].append(-1) # after arriving at dest, no action should be taken
    spt[loc_r][loc_c][1] = 0
    
    q = Queue()
    q.put([loc, 0])
    
    while not q.empty():
        [(row, col), cur_dist] = q.get() # current location we are exploring the neighbors of
        for action in ['N', 'E', 'S', 'W']: 
            if action == 'S':
                (next_r, next_c) = (min(row + 1, max_row), col)
                if spt[next_r][next_c][1] >= cur_dist + 1 and desc[next_r][next_c]!= obstacle_code:
                    if spt[next_r][next_c][1] > cur_dist + 1:  
                        spt[next_r][next_c][1] = cur_dist + 1
                        q.put([(next_r, next_c), cur_dist + 1])
                    spt[next_r][next_c][0].append((-1,0)) # add to optimal actions list
                                     
            elif action == 'N':
                (next_r, next_c) = (max(row - 1, 0), col)
                if spt[next_r][next_c][1] >= cur_dist + 1 and desc[next_r][next_c]!= obstacle_code:
                    if spt[next_r][next_c][1] > cur_dist + 1:   
                        spt[next_r][next_c][1] = cur_dist + 1
                        q.put([(next_r, next_c), cur_dist + 1])
                    spt[next_r][next_c][0].append((1,0))
                        
            elif action == 'E':
                (next_r, next_c) = (row, min(col + 1, max_col))
                
                if spt[next_r][next_c][1] >= cur_dist + 1 and desc[next_r][next_c]!= obstacle_code:
                    if spt[next_r][next_c][1] > cur_dist + 1:   
                        spt[next_r][next_c][1] = cur_dist + 1
                        q.put([(next_r, next_c), cur_dist + 1])
                    spt[next_r][next_c][0].append((0,-1))
            
            elif action == 'W':
                (next_r, next_c) = (row, max(col - 1, 0))
                
                if spt[next_r][next_c][1] >= cur_dist + 1 and desc[next_r][next_c]!= obstacle_code:
                    if spt[next_r][next_c][1] > cur_dist + 1:
                        spt[next_r][next_c][1] = cur_dist + 1
                        q.put([(next_r, next_c), cur_dist + 1])
                    spt[next_r][next_c][0].append((0,1))

    return spt