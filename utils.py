import numpy as np

def process_grids(observation, env_params, visual=True):
    agent_grid, static_grid, dynamic_grid = observation
    if visual:
        agent_static_grid_processed, dynamic_grid_processed = np.array((N, M, 3)), np.array((N, M, 3))
    else:
        agent_static_grid_processed, dynamic_grid_processed = np.array((3, N, M)), np.array((3, N, M))
    
    def set_channel(a, b, grid, color):
        for channel in range(3):
            grid[channel][a][b] = color[channel]
    
    for i in env_params.N:
        for j in env_params.M:
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
                dynamic_grid_processed = [greyscale_val]*3
            else:
                set_channel(i, j, dynamic_grid_processed, [greyscale_val]*3)
    return (agent_static_grid_processed, dynamic_grid_processed)