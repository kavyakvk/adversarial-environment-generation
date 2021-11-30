import agent
import utils
import environment

import copy
import pickle
import torch
import numpy as np

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

def train(grid, agents, env_params, num_episodes=50):
    env = environment.Environment(env_params, grid)

    for episode in range(num_episodes):
        print("Episode", episode)
        # Initialize the environment and spawn queue
        env.reset(grid)
        env.initialize_spawn_queue(agents)
        #old_observations = [prepare_observation(env.get_empty_observation()) for agent in agents]
        observations = None
        
        for step in range(env_params['steps']):
            # Spawn agents if necessary
            env.spawn_agents(agents)

            # Get observation for each agent and calculate an action for each agent
            observations = env.update_observation(agents)
            for agent_idx in range(len(agents)):
                agent = agents[agent_idx]
                obs = observations[agent_idx]
                observations[agent_idx] = utils.prepare_observation(obs, env_params, (agent.screen_height, agent.screen_width))
            
            actions = []
            for agent_idx in range(len(agents)):
                agent = agents[agent_idx]
                state = observations[agent_idx]#-old_observations[agent_idx]
                action = agent.get_action(state, env.get_valid_movements(agent), train=True)
                actions.append(action)
            actions_tensor = torch.tensor(actions, device=DEVICE)
            
            # Step actions in the environment
            environment_actions = []
            for agent_idx in range(len(agents)):
                if agents[agent_idx].food:
                    environment_actions.append((env.environment_actions[action], env_params['pheromone']['step']))
                else:
                    environment_actions.append((env.environment_actions[action], env_params['pheromone']['step']))
            rewards = env.step(agents, actions)
            rewards_tensor = torch.tensor(rewards, device=DEVICE)

            # Get new observation for each agent
            next_observations = [prepare_observation(obs, env_params, (agent.screen_height, agent.screen_width)) for obs in env.update_observation(agents)]

            # Store the transitions in memory
            for agent_idx in range(len(agents)):
                agent = agents[agent_idx]
                state = observations[agent_idx]#-old_observations[agent_idx]
                next_state = next_observations[agent_idx]#-observations[agent_idx]
                agent.memory.push(state, actions_tensor[agent_idx], next_state, rewards_tensor[agent_idx])

            # Save observations
            #old_observations = observations

            # Perform one step of the optimization (on the policy network) for all agents
            for agent in agents:
                agent.optimize_model()

        for agent in agents:
            # Update the target network, copying all weights and biases in DQN
            if episode % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

def dqn_main():
    agents = [agent.DQNAgent(i, ENV_PARAMS) for i in range(5)]
    grid = utils.generate_random_grid(ENV_PARAMS)
    train(grid, agents, ENV_PARAMS, num_episodes=50)


dqn_main()