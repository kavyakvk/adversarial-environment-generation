import abc
from utils import *
import torch

class Agent():
    def __init__(self, id):
        self.location = (0,0)
        self.prev_location = (0,0)
        self.food = 0
        self.active = 0
        self.id = id
        self.bfs_active = 0
    
    def get_state(self):
        return self.location, self.food
    
    @abc.abstractmethod
    def get_action(self, observation, valid_movements):
        # Returns movement, pheromone
        pass 

    def __str__(self):
        return ''+ str(self.location) + ' ' + str(self.active) + ' ' + str(self.food)

class RandomAgent(Agent):
    def get_action(self, observation, valid_movements):
        return random.choice(valid_movements), random.random()
    
    
class SwarmAgent(Agent):
    def __init__(self, id, env_params, spt):
        self.env_params = env_params
        self.spt = spt
        self.obs_rad = self.env_params['observation_radius']
        obs_window = self.env_params['observation_radius']*2+1
        super().__init__(id)
        
    def get_action(self, observation, valid_movements):
        agent_grid, static_grid, dynamic_grid = observation
        last_action = tuple(np.array(self.location) - np.array(self.prev_location))
        if self.food:
            # choose direction with BFS
            next_movement = np.random.choice(spt[self.x][self.y][0])
        
        else:
            # head toward high pheromone
            if last_action == (-1,0):
                d = dynamic_grid[:self.obs_rad + 1, :]

            elif last_action == (0,1):
                d = dynamic_grid[:, self.obs_rad:]

            elif last_action == (1,0):
                d = dynamic_grid[self.obs_rad:, :]

            elif last_action == (0,-1):
                d = dynamic_grid[:, :self.obs_rad + 1]
            else: 
                d = dynamic_grid
        
        best_loc = np.unravel_index(d.argmax(), d.shape)
                
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

        # if list empty take random action from all actions not blocked by obstacles
        good_actions = list(set(good_actions).intersection(set(valid_movements)))
        if len(good_actions) > 0:
            next_movement = good_actions[np.random.randint(len(good_actions))]
        else:
            next_movement = valid_movements[np.random.randint(len(valid_movements))]
            
        return next_movement, self.env_params['pheromone']['step']



class DQNAgent(Agent):
    def __init__(self, id, env_params):
        self.env_params = self.env_params
        obs_window = self.env_params['observation_radius']*2+1
        self.input_shape = (obs_window, obs_window)

        # construct the model
        agent_static = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        agent_dynamic = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        super(self)
    
    def get_action(self, observation, valid_movements):
        agent_static_grid, dynamic_grid = utils.process_grids(observation, visual=False)
        


