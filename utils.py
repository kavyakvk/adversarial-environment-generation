def process_grids(observation, env_params):
    agent_grid, static_grid, dynamic_grid = observation
    agent_static_grid_processed, dynamic_grid_processed = np.array((N, M, 3)), np.array((N, M, 3))
    for i in self.env_params.N:
        for j in self.env_params.M:
            # combine agent and static grids
            if static_grid[i,j] == self.env_params['coding_dict']['empty'] and agent_grid[i,j] == self.env_params['coding_dict']['agent']:
                agent_static_grid_processed[i,j] = self.env_params['rgb_coding'][self.env_params['coding_dict']['agent']]
            else:
                agent_static_grid_processed[i,j] = self.env_params['rgb_coding'][static_grid[i,j]]
            # compute greyscale values for dynamic grid
            greyscale_val = 255 * (self.env_params['pheromone']['cap'] - dynamic_grid[i,j])/self.env_params['pheromone']['cap']
            assert(greyscale_val > 0 and greyscale_val <= 255)
            dynamic_grid_processed = [greyscale_val]*3
    return (agent_static_grid_processed, dynamic_grid_processed)