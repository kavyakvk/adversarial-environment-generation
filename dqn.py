from agent import Agent
from utils import *
import torch

class DQNAgent(Agent):
    def __init__(self, id, env_params):
        self.env_params = self.env_params
        obs_window = self.env_params['observation_radius']*2+1
        self.input_shape = (obs_window, obs_window)
        super(self)
    
    def get_action(self, observation, valid_movements):
        agent_static_grid, dynamic_grid = utils.process_grids()

        
            
        

        

