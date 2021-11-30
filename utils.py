import numpy as np
from queue import Queue
import random
import copy

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


# get array of movements to take from each square towards hive
# avoids obstacles (coded as 4 by default)
# desc: static grid; loc: use (0,0) to go to hive
def expert_navigation_policy_set(desc, loc, obstacle_code = 4):
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