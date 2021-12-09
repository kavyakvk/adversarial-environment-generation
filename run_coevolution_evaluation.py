import agent

import copy
import pickle
import argparse
import os
import random
from tqdm import tqdm
import utils

ENV_PARAMS = {'coding_dict': {'empty': 0, 'agent': 1, 'bounds': 2, 'hive': 3, 'blockade': 4, 'food_start': 6}, 
                            'N': 12, 'M': 12, 'max_food': 5, 'observation_radius': 1, 'steps': 300, 'spawn_rate': 2, 
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

ENV_PARAMS['grid']['food'] = 40
ENV_PARAMS['grid']['blockade'] = 20

def get_food_collected(env_params, agents, grid):
    env = environment.Environment(env_params, grid)
    for agent in agents:
        agent.set_spt(env.spt)
    food_collected = env.run_episode(agents, visualize=False)
    return food_collected

run_folder = f'Pickled/Final/KavyaRuns/Coevolution/Coevolution5DQNAgent_20DUELiterations_5GAiterations_30GApopulation_20AGENTepisodes/'

pickle_dict = None
grids, fitness_values = None, None
with open("Pickled/Final/KavyaRuns/GA_5DQNAgent_2tile_0.1mutation_40food_20blocks", 'rb') as f:
    pickle_dict = pickle.load(f)
grids, fitness_values = pickle_dict['grids'], pickle_dict['fitness values']

selected = sorted(range(len(fitness_values[-1])), key=lambda x: fitness_values[-1][x], reverse=True)[:10]
elitism_population = [grids[-1][x] for x in selected]
middle_population = [x for x in range(len(fitness_values)) if x not in selected]
middle_population = [grids[-1][x] for x in middle_population]
middle_population = random.sample(middle_population, 10)
random_population = utils.generate_n_valid_feasible_grids(10, ENV_PARAMS)

test_grids = random_population+middle_population+elitism_population

duel_evaluation_filename = f'{run_folder}Coevolution_Evaluation'
if os.path.exists(duel_evaluation_filename):
    duel_evaluation_pickle = pickle.load(open(duel_evaluation_filename, 'rb'))
else:
    duel_evaluation_pickle = {}

iterations = len(os.listdir(run_folder))-2
for iteration in range(args.duel_train_iterations):
    if iteration not in duel_evaluation_pickle:
        print("iteration", iteration)
        duel_evaluation_pickle[iteration] = {}

        filename = f'{run_folder}target_net_{iteration}iteration.pt'
        test_agents = [agent.DQNAgent(i, ENV_PARAMS, net_filepath=filename) for i in range(5)]

        for grid_idx in tqdm(range(len(test_grids))):
            grid = test_grids[grid_idx]
            duel_evaluation_pickle[iteration][grid_idx] = get_food_collected(ENV_PARAMS, agents, grid)
        print(duel_evaluation_pickle[iteration])
        with open(duel_evaluation_filename, 'wb') as f:
            pickle.dump(duel_evaluation_pickle, f)