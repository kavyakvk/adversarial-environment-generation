import agent
import utils
import environment

import copy
import pickle
import torch
import numpy as np
from tqdm import tqdm
import random
from collections import namedtuple, deque

ENV_PARAMS = {'coding_dict': {'empty': 0, 'agent': 1, 'bounds': 2, 'hive': 3, 'blockade': 4, 'food_start': 5}, 
                            'N': 10, 'M': 10, 'max_food': 5, 'observation_radius': 1, 'steps': 300, 'spawn_rate': 2, 
                            'pheromone': {'evaporation': 0.05, 'diffusion': 0.1, 'step': 0.1, 'step_if_food': 0.3, 'cap': 5}, 
                            'grid': {'food': 40, 'blockade': 20}, 
                            'env_actions': [(0,0),(0,-1), (0,1), (1,0), (-1,0)],
                            'rgb_coding': {0: [0, 0, 0], 
                                            1: [150, 0, 150], 2: [100, 100, 100], 
                                            3: [150, 150, 0], 4: [45, 0, 255], 
                                            5: [0, 255, 45], 6: (0, 250, 50), 
                                            7: (0, 245, 55), 8: (0, 240, 60), 
                                            9: (0, 235, 65), 10: (0, 230, 70), 
                                            11: (0, 225, 75), 12: (0, 220, 80), 
                                            13: (0, 215, 85), 14: (0, 210, 90)}}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train(grid, agents, env_params, filename, num_episodes=50):
    episode_loss = [0]
    episode_rewards = [[0 for agent in agents]]

    shared_memory = ReplayMemory(10000) 

    train_agent = agents[0]

    for episode in range(num_episodes):
        env = environment.Environment(env_params, grid)

        for agent in agents:
            agent.set_spt(env.spt)
        
        print("Episode", episode)
        # Initialize the environment and spawn queue
        env.reset(agents, grid)
        env.initialize_spawn_queue(agents)
        #old_observations = [prepare_observation(env.get_empty_observation()) for agent in agents]
        observations = None
        
        for step in tqdm(range(env_params['steps'])):
            # Spawn agents if necessary
            env.spawn_agents(agents)

            # Get observation for each agent and calculate an action for each agent
            observations = env.update_observation(agents)
            for agent_idx in range(len(agents)):
                agent = agents[agent_idx]
                obs = observations[agent_idx]
                observations[agent_idx] = utils.prepare_observation(obs, env_params, (agent.screen_height, agent.screen_width))
            
            movement_actions = []
            environment_actions = []
            for agent_idx in range(len(agents)):
                agent = agents[agent_idx]
                state = observations[agent_idx]#-old_observations[agent_idx]
                movement, pheromone = agent.get_action(state, env.get_valid_movements(agent), train=True)
                movement_actions.append(movement)
                environment_actions.append((env_params['env_actions'][movement.item()], pheromone))
            movement_actions_tensor = torch.tensor(movement_actions, device=DEVICE)

            rewards = env.step(agents, environment_actions)
            rewards_tensor = torch.tensor(rewards, device=DEVICE)
            episode_rewards[episode] = [episode_rewards[episode][k]+rewards[k] for k in range(len(agents))]

            # Get new observation for each agent
            next_observations = [utils.prepare_observation(obs, env_params, (agent.screen_height, agent.screen_width)) for obs in env.update_observation(agents)]

            # Store the transitions in memory
            for agent_idx in range(len(agents)):
                state = observations[agent_idx]#-old_observations[agent_idx]
                next_state = next_observations[agent_idx]#-observations[agent_idx]
                shared_memory.push(state, movement_actions_tensor[agent_idx].view(1,1), next_state, rewards_tensor[agent_idx].view(1))

            # Save observations
            #old_observations = observations

            # Perform one step of the optimization (on the policy network) for all agents
            loss = train_agent.optimize_model(shared_memory)
            if loss is not None:
                    episode_loss[episode] += loss/env_params['steps']
                
        print(episode_loss[-1])
        episode_loss.append(0)
        episode_rewards.append([0 for agent in agents])
        
        for agent in agents:
            # Update the target network, copying all weights and biases in DQN
            if episode % agent.TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(train_agent.policy_net.state_dict())
        
        torch.save(train_agent.target_net.state_dict(), filename)
        
    return episode_loss, episode_rewards 

def dqn_main():
    agents = [agent.DQNAgent(i, ENV_PARAMS) for i in range(5)]
    grid = utils.generate_random_grid(ENV_PARAMS)
    episode_loss, episode_rewards = train(grid, agents, ENV_PARAMS, filename="DQN/target_net.pt", num_episodes=50)

    with open('Pickled/DQN_training_rewards.pkl', 'wb') as f:
        pickle.dump(episode_rewards, f)
    with open('Pickled/DQN_training_loss.pkl', 'wb') as f:
        pickle.dump(episode_loss, f)


dqn_main()