import numpy as np
import random
import copy
from utils import *

ENV_PARAMS = {'coding_dict': {'empty': 0, 'agent': 1, 'bounds': 2, 'hive': 3, 'blockade': 4, 'food_start': 5}, 
                            'N': 20, 'M': 20, 'max_food': 5, 'observation_radius': 5, 'steps': 5000, 'spawn_rate': 2, 
                            'pheromone': {'evaporation': 0.1, 'diffusion': 0.1, 'step': 0.2, 'cap': 5}, 
                            'grid': {'food': 40, 'blockade': 20}, 
                            'rgb_coding': {0: [0, 0, 0], 1: [150, 0, 150], 2: [100, 100, 100], 3: [150, 150, 0], 4: [45, 0, 255], 5: [0, 255, 45], 6: (0, 250, 50), 7: (0, 245, 55), 8: (0, 240, 60), 9: (0, 235, 65), 10: (0, 230, 70), 11: (0, 225, 75), 12: (0, 220, 80), 13: (0, 215, 85), 14: (0, 210, 90)}}

class Environment:
    def __init__(self, env_params, grid=None):
        '''
            Grid
        '''
        self.rows = env_params['N']
        self.cols = env_params['M']
        self.agent_grid = np.zeros((self.rows, self.cols), dtype=float)     # one-hot encoding of agent locations
        self.agent_nums = np.zeros((self.rows, self.cols), dtype=float)     # number of agents in each location
        self.dynamic_grid = np.zeros((self.rows, self.cols), dtype=float)  # pheromone values for every location in grid

        if grid is not None:        # if there is a grid passed through
            self.static_grid = grid
        else:                       # for testing (if a grid isn't provided)
            self.static_grid = np.zeros((self.rows, self.cols), dtype=float)
            self.static_grid[0][0] = env_params['coding_dict']['hive']
            self.static_grid[self.rows-1][self.cols-1] = 9              # Place food in the corner
        
        '''
            BFS
        '''
        self.spt = [[q[0] for q in r] for r in expert_navigation_policy_set(self.static_grid, (0,0))]
        
        '''
            Params
        '''
        self.evaporation_rate = env_params['pheromone']['evaporation']
        self.observation_radius = env_params['observation_radius']
        self.total_food = 0
        self.time_step = 0
        self.total_steps = env_params['steps']
        self.spawn_rate = env_params['spawn_rate']
        self.env_params = env_params

        '''
            Spawn queue
        '''
        self.spawn_queue = []


    def get_valid_movements(self, agent):
        location, food = agent.get_state()
        pos_x, pos_y = location
        possible_movements = [(0,-1), (0,1), (1,0), (-1,0)]
        valid_movements = [(0,0)]
        for movement in possible_movements:
            new_location = (pos_x + movement[0], pos_y + movement[1])
            if new_location[0] < self.rows and new_location[0] >= 0 and new_location[1] < self.cols and new_location[1] >= 0 and self.static_grid[new_location[0]][new_location[1]] not in [self.env_params['coding_dict']['bounds'], self.env_params['coding_dict']['blockade']]:
                valid_movements.append(movement)
        if len(valid_movements) == 0:
            print(self.static_grid)
        assert(len(valid_movements) > 0)
        return valid_movements
    
    def step(self, agents, actions=None):
        self.time_step += 1

        # Pheromone evaporation
        for i in range(self.rows):
            for j in range(self.cols):
                self.dynamic_grid[i, j] *= (1 - self.evaporation_rate)

        # Update for each active agent
        for agent_idx in agents:
            agent = agents[agent_idx]
            if agent.active == 1:
                if ((actions is not None) and (actions[agent_idx] is not None)):
                    movement, pheromone = actions[agent_idx]
                else:
                    movement, pheromone = agent.get_action(self.get_observation(agent.location), self.get_valid_movements(agent))
                location, food = agent.get_state()
                new_location = (location[0] + movement[0], location[1] + movement[1])
                # Add pheromone
                self.dynamic_grid[location] = min(self.env_params['pheromone']['cap'], self.dynamic_grid[location] + pheromone)       # cap pheromone
                # Update agent location
                self.agent_nums[location] -= 1
                if self.agent_nums[location] == 0:
                    self.agent_grid[location] = 0   # Delete previous location from static
                if agent.prev_location != location:
                    agent.prev_location = location
                self.agent_nums[new_location] += 1
                self.agent_grid[new_location] = 1   # Add new location to static
                agent.location = new_location
                # If agent is now at food and doesn't already have food
                if self.static_grid[new_location[0]][new_location[1]] >= self.env_params['coding_dict']['food_start'] and agent.food == 0:
                    agent.food = 1
                    if self.static_grid[new_location[0]][new_location[1]] == self.env_params['coding_dict']['food_start']:
                        self.static_grid[new_location[0]][new_location[1]] == 0    # now no more food
                    else:
                        self.static_grid[new_location[0]][new_location[1]] -= 1    # decrement food by 1
                    agent.bfs_active = 1        # activate bfs
                # If agent is now at hive
                if self.static_grid[new_location[0]][new_location[1]] == self.env_params['coding_dict']['hive']:
                    agent.active = 0
                    self.spawn_queue.append(agent.id)
                    self.total_food += agent.food
                    agent.food = 0

    def get_observation(self, location):
        # Returns partially observable observation centered around location
        observation_agent = np.zeros((2*self.observation_radius + 1, 2*self.observation_radius + 1), dtype=float)
        observation_grid = np.zeros((2*self.observation_radius + 1, 2*self.observation_radius + 1), dtype=float)
        observation_dynamic = np.zeros((2*self.observation_radius + 1, 2*self.observation_radius + 1), dtype=float)

        upper_left_loc = (location[0] - self.observation_radius, location[1] - self.observation_radius)
        for i in range(self.observation_radius * 2 + 1):
            for j in range(self.observation_radius * 2 + 1):
                temp_loc = (upper_left_loc[0] + i, upper_left_loc[0] + j)
                if temp_loc[0] < 0 or temp_loc[1] < 0 or temp_loc[0] >= self.rows or temp_loc[1] >= self.cols:      # if out of bounds
                    observation_agent[i][j] = 0
                    observation_grid[i][j] = self.env_params['coding_dict']['bounds']
                    observation_dynamic[i][j] = 0
                else:
                    observation_agent[i][j] = self.agent_grid[temp_loc[0]][temp_loc[1]]
                    observation_grid[i][j] = self.static_grid[temp_loc[0]][temp_loc[1]]
                    observation_dynamic[i][j] = self.dynamic_grid[temp_loc[0]][temp_loc[1]]


        # observation_grid_static = np.add(self.static_grid[location[0]-self.observation_radius : location[0]+self.observation_radius+1][location[1]-self.observation_radius : location[1]+self.observation_radius+1], \
        #                             self.agent_grid[location[0]-self.observation_radius : location[0]+self.observation_radius+1][location[1]-self.observation_radius : location[1]+self.observation_radius+1])
        # observation_dynamic = self.dynamic_grid[location[0]-self.observation_radius : location[0]+self.observation_radius+1][location[1]-self.observation_radius : location[1]+self.observation_radius+1]
        return observation_agent, observation_grid, observation_dynamic

    def reset(self, grid=None):
        self.agent_grid = np.zeros((self.rows, self.cols), dtype=float)     # one-hot encoding of agent locations
        self.agent_nums = np.zeros((self.rows, self.cols), dtype=float)
        self.dynamic_grid = np.zeros((self.rows, self.cols), dtype=float)  # pheromone values for every location in grid

        if grid is not None:        # if there is a grid passed through
            self.static_grid = grid
        else:
            self.static_grid = np.zeros((self.rows, self.cols), dtype=float)
            self.static_grid[0][0] = self.env_params['coding_dict']['hive']
            self.static_grid[self.rows-1][self.cols-1] = 9              # Place food in the corner
        
        
        '''
            Params
        '''
        self.total_food = 0
        self.time_step = 0

        '''
            Spawn queue
        '''
        self.spawn_queue = []
    
    def spawn_agents(self, agents):
        for i in range(min(self.spawn_rate, len(self.spawn_queue))):
            new_agent = agents[self.spawn_queue.pop(0)]
            new_agent.active = 1
            new_agent.location = (1,0)      # spawn agents next to hive
            self.agent_grid[1][0] = 1
            self.agent_nums[1][0] += 1
            # print(agents[0])
    
    def update_observation(self, agents):
        for agent in agents:
            if agent.active == 1:
                agent.observation = self.get_observation(agent.location)
                # self.visualize_map(agent.observation[0])

    def run_episode(self, agents, grid=None, visualize=False):
        # Add all agents to spawn queue
        self.spawn_queue = [i for i in range(len(agents))]  
        
        env_observations = [(self.agent_grid.copy(), self.static_grid.copy(), self.dynamic_grid.copy())]

        # Run through time steps
        for time_step in range(self.total_steps):
            # Spawn agents if they can be spawned
            self.spawn_agents(agents)

            # Update observation for every active agent
            self.update_observation(agents)
                    
            # Environment.step
            self.step(agents)
            
            # Save observation for visualization
            if visualize:
                env_observations.append((self.agent_grid.copy(), self.static_grid.copy(), self.dynamic_grid.copy()))
        
        food_collected = self.total_food

        if visualize:
            print('grid')
            self.visualize_map(self.static_grid)
            print('static')
            self.visualize_map(self.agent_grid)
        
        # Reset environment
        self.reset(grid)
        
        # Return amount of collected food
        if visualize:
            return food_collected, env_observations
        else:
            return food_collected

    def visualize_map(self, np_array):
        # Print out np_array
        print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in np_array]))

