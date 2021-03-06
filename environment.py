import numpy as np
import random
import copy
import utils

class Environment:
    def __init__(self, env_params, grid=None, assert_food_blockade_match = True):
        '''
            Grid
        '''
        self.rows = env_params['N']
        self.cols = env_params['M']
        self.agent_grid = np.zeros((self.rows, self.cols), dtype=float)     # one-hot encoding of agent locations
        self.agent_nums = np.zeros((self.rows, self.cols), dtype=float)     # number of agents in each location
        self.dynamic_grid = np.zeros((self.rows, self.cols), dtype=float)  # pheromone values for every location in grid

        if grid is not None:        # if there is a grid passed through
            self.static_grid = copy.deepcopy(grid)
        else:                       # for testing (if a grid isn't provided)
            self.static_grid = np.zeros((self.rows, self.cols), dtype=float)
            self.static_grid[0][0] = env_params['coding_dict']['hive']
            self.static_grid[1][0] = self.env_params['coding_dict']['hive']
            self.static_grid[0][1] = self.env_params['coding_dict']['hive']
            self.static_grid[self.rows-1][self.cols-1] = 9              # Place food in the corner
        if assert_food_blockade_match:
            utils.check_valid(self.static_grid, env_params)
        self.environment_actions = env_params['env_actions']

        '''
            BFS
        '''
        self.spt = [[q[0] for q in r] for r in utils.expert_navigation_policy_set(desc=self.static_grid, loc=(0,0), env_params=env_params)]
        
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
            if ((new_location[0] < self.rows and new_location[0] >= 0) and 
                (new_location[1] < self.cols and new_location[1] >= 0) and 
                (self.static_grid[new_location[0]][new_location[1]] != self.env_params['coding_dict']['bounds']) and 
                (self.static_grid[new_location[0]][new_location[1]] != self.env_params['coding_dict']['blockade'])):
                valid_movements.append(movement)
        if len(valid_movements) == 0:
            print(self.static_grid)
        assert(len(valid_movements) > 0)
        return valid_movements
    
    def step(self, agents, actions=None):
        self.time_step += 1
        step_rewards = [0 for agent in range(len(agents))]
        collected_food = 0
        
        # Pheromone evaporation
        for i in range(self.rows):
            for j in range(self.cols):
                self.dynamic_grid[i, j] *= (1 - self.evaporation_rate)

        # Update for each active agent
        for agent_idx in range(len(agents)):
            agent = agents[agent_idx]
            if agent.active == 1:
                agent_observation = self.get_observation(agent.location)
                if ((actions is not None) and (actions[agent_idx] is not None)):
                    movement, pheromone = actions[agent_idx]
                else:
                    movement, pheromone = agent.get_action(agent_observation, self.get_valid_movements(agent))
                
                # REWARD if agent can see food
                if np.any(np.isin(np.arange(self.env_params['coding_dict']['food_start'], 
                                            self.env_params['coding_dict']['food_start']+self.env_params['max_food']),
                                    agent_observation[1])):
                    step_rewards[agent_idx] += 0.1

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
                        self.static_grid[new_location[0]][new_location[1]] = self.env_params['coding_dict']['empty']    # now no more food
                    else:
                        self.static_grid[new_location[0]][new_location[1]] -= 1    # decrement food by 1
                    agent.bfs_active = 1        # activate bfs
                    # REWARD for agent picking up food
                    step_rewards[agent_idx] += 5
                # If agent is now at hive
                if self.static_grid[new_location[0]][new_location[1]] == self.env_params['coding_dict']['hive']:
                    agent.active = 0
                    self.spawn_queue.append(agent.id)
                    self.total_food += agent.food
                    if agent.food == 1:
                        # REWARD for agent returning food
                        step_rewards[agent_idx] += 2
                        collected_food += 1
                    agent.food = 0
        return step_rewards, collected_food
    
    def get_empty_observation(self):
        observation_agent = np.zeros((2*self.observation_radius + 1, 2*self.observation_radius + 1), dtype=float)
        observation_grid = np.zeros((2*self.observation_radius + 1, 2*self.observation_radius + 1), dtype=float)
        observation_dynamic = np.zeros((2*self.observation_radius + 1, 2*self.observation_radius + 1), dtype=float)
        return observation_agent, observation_grid, observation_dynamic 

    def get_observation(self, location):
        # Returns partially observable observation centered around location
        observation_agent, observation_grid, observation_dynamic = self.get_empty_observation()

        upper_left_loc = (location[0] - self.observation_radius, location[1] - self.observation_radius)
        for i in range(self.observation_radius * 2 + 1):
            for j in range(self.observation_radius * 2 + 1):
                temp_loc = (upper_left_loc[0] + i, upper_left_loc[1] + j)
                if temp_loc[0] < 0 or temp_loc[1] < 0 or temp_loc[0] >= self.rows or temp_loc[1] >= self.cols:      # if out of bounds
                    # observation_agent[i][j] = 0
                    observation_grid[i][j] = self.env_params['coding_dict']['bounds']
                    # observation_dynamic[i][j] = 0
                else:
                    observation_agent[i][j] = self.agent_grid[temp_loc[0]][temp_loc[1]]
                    observation_grid[i][j] = self.static_grid[temp_loc[0]][temp_loc[1]]
                    observation_dynamic[i][j] = self.dynamic_grid[temp_loc[0]][temp_loc[1]]


        # observation_grid_static = np.add(self.static_grid[location[0]-self.observation_radius : location[0]+self.observation_radius+1][location[1]-self.observation_radius : location[1]+self.observation_radius+1], \
        #                             self.agent_grid[location[0]-self.observation_radius : location[0]+self.observation_radius+1][location[1]-self.observation_radius : location[1]+self.observation_radius+1])
        # observation_dynamic = self.dynamic_grid[location[0]-self.observation_radius : location[0]+self.observation_radius+1][location[1]-self.observation_radius : location[1]+self.observation_radius+1]
        return observation_agent, observation_grid, observation_dynamic

    def reset(self, agents, grid=None):
        self.agent_grid = np.zeros((self.rows, self.cols), dtype=float)     # one-hot encoding of agent locations
        self.agent_nums = np.zeros((self.rows, self.cols), dtype=float)
        self.dynamic_grid = np.zeros((self.rows, self.cols), dtype=float)  # pheromone values for every location in grid

        if grid is not None:        # if there is a grid passed through
            self.static_grid = copy.deepcopy(grid)
        else:
            self.static_grid = np.zeros((self.rows, self.cols), dtype=float)
            self.static_grid[0][0] = self.env_params['coding_dict']['hive']
            self.static_grid[1][0] = self.env_params['coding_dict']['hive']
            self.static_grid[0][1] = self.env_params['coding_dict']['hive']
            self.static_grid[self.rows-1][self.cols-1] = 9              # Place food in the corner
        
        '''
            Reset Agents
        '''
        for agent in agents:
            agent.food = 0
            agent.active = 0
            agent.bfs_active = 0
            agent.location = (0,0)
            agent.prev_location = (0,0)
        '''
            Params
        '''
        self.total_food = 0
        self.time_step = 0

        '''
            Spawn queue
        '''
        self.spawn_queue = []
    
    def initialize_spawn_queue(self, agents):
        self.spawn_queue = [i for i in range(len(agents))]  
    
    def spawn_agents(self, agents):
        for i in range(min(self.spawn_rate, len(self.spawn_queue))):
            new_agent = agents[self.spawn_queue.pop(0)]
            new_agent.active = 1
            new_agent.location = random.choice([(1,0),(0,1)])     # spawn agents next to hive or below hive
            self.agent_grid[1][0] = 1
            self.agent_nums[1][0] += 1
    
    def update_observation(self, agents):
        observations = []
        for agent in agents:
            if agent.active == 1:
                obs = self.get_observation(agent.location)
                agent.observation = obs
                # self.visualize_map(agent.observation[0])
                observations.append(obs)
            else:
                observations.append(self.get_empty_observation())
        return observations


    def run_episode(self, agents, grid=None, visualize=False):
        # Add all agents to spawn queue
        self.initialize_spawn_queue(agents)
        
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
        self.reset(agents, grid)
        
        # Return amount of collected food
        if visualize:
            return food_collected, env_observations
        else:
            return food_collected

    def visualize_map(self, np_array):
        # Print out np_array
        print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in np_array]))

