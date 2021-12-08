import agent
import utils
import environment

import copy
import pickle
import torch
import numpy as np
from tqdm import tqdm
import random
import os
from collections import namedtuple, deque

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

def train(agents, env_params, filename, num_episodes=50, grids=None, random_proportion=0.2, verbose=True):
    tqdm_disable = not verbose
    gpu_device = agents[0].gpu

    episode_loss = [0]
    episode_rewards = [[0 for agent in agents]]
    episode_food = [0]

    shared_memory = ReplayMemory(10000) 

    train_agent = agents[1]

    chosen_grids = grids
    if grids is None:
        chosen_grids = utils.generate_n_valid_feasible_grids(num_episodes, env_params)
    else:
        num_random_grids = int(random_proportion*num_episodes)
        num_elite_grids = min(len(grids), num_episodes-num_random_grids)
        num_random_grids = num_episodes - num_elite_grids
        chosen_grids = random.sample(grids, num_elite_grids)
        chosen_grids.extend(utils.generate_n_valid_feasible_grids(num_random_grids, env_params))
        random.shuffle(chosen_grids)
        
    for episode in range(num_episodes):
        grid = chosen_grids[episode]
        env = environment.Environment(env_params, grid)

        for agent in agents:
            agent.set_spt(env.spt)
        
        if verbose:
            print("Episode", episode)
        # Initialize the environment and spawn queue
        env.reset(agents, grid)
        env.initialize_spawn_queue(agents)
        #old_observations = [prepare_observation(env.get_empty_observation()) for agent in agents]
        observations = None
        
        for step in tqdm(range(env_params['steps']), disable=tdqm_disable):
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
            movement_actions_tensor = torch.tensor(movement_actions, device=gpu_device)

            rewards, collected_food = env.step(agents, environment_actions)
            rewards_tensor = torch.tensor(rewards, device=gpu_device)
            episode_rewards[episode] = [episode_rewards[episode][k]+rewards[k] for k in range(len(agents))]
            episode_food[episode] += collected_food

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
                
        print(episode_loss[-1], episode_rewards[-1], episode_food[-1])
        episode_loss.append(0)
        episode_rewards.append([0 for agent in agents])
        episode_food.append(0)
        
        for agent in agents:
            # Update the target network, copying all weights and biases in DQN
            if episode % agent.TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(train_agent.policy_net.state_dict())
        
        torch.save(train_agent.target_net.state_dict(), filename)
        
    return episode_loss, episode_rewards 

def dqn_main(env_params, agents, grids = None, random_proportion=0.2, filename=cwd+"DQN/target_net.pt", num_episodes=20, verbose=True):
    episode_loss, episode_rewards = train(agents, env_params, 
                                            filename=filename, num_episodes=num_episodes, 
                                            grids=grids, random_proportion=0.2,
                                            verbose=True)

    # with open(cwd+'DQN/DQN_training_rewards.pkl', 'wb') as f:
    #     pickle.dump(episode_rewards, f)
    # with open(cwd+'DQN/DQN_training_loss.pkl', 'wb') as f:
    #     pickle.dump(episode_loss, f)
    return episode_rewards, episode_loss