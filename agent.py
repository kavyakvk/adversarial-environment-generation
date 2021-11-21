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
