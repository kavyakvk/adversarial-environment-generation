import abc
from utils import *
import torch

class Agent():
    def __init__(self, id):
        self.location = (0,0)
        self.prev_location = (0,0)
        self.food = 0
        self.active = 0
        self.id = None
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


class DQNAgent(Agent):
    def __init__(self, id, env_params):
        self.env_params = self.env_params
        obs_window = self.env_params['observation_radius']*2+1
        self.input_shape = (obs_window, obs_window)
        super(self)
    
    def get_action(self, observation, valid_movements):
        agent_static_grid, dynamic_grid = utils.process_grids(observation)
