from geneticalg import GeneticAlgorithm
import agent
import train_dqn

import copy
import pickle
import argparse
import os

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_food', default=40, type=int)
    parser.add_argument('--num_blockade', default=20, type=int)

    parser.add_argument('--num_agents', default=5, type=int)
    parser.add_argument('--agent_gpu', default=-1, type=int)
    parser.add_argument('--agent_initial_weights', default="DQN/target_net.pt", type=str)

    parser.add_argument('--ga_population_size', default=5, type=int)
    parser.add_argument('--ga_rate_elitism', default=0.2, type=float)
    parser.add_argument('--ga_rate_mutation', default=0.1, type=float)
    parser.add_argument('--ga_iterations', default=50, type=float)
    parser.add_argument('--ga_tile_size', default=2, type=int)
    parser.add_argument('--ga_initial_population', default="Pickled/Final/KavyaRuns/GA_5DQNAgent_2tile_0.1mutation_40food_20blocks", type=str)

    parser.add_argument('--duel_train_iterations', default=5, type=int)

    args = parser.parse_args()

    # Override global environment parameters
    ENV_PARAMS['grid']['food'] = args.num_food
    ENV_PARAMS['grid']['blockade'] = args.num_blockade

    ga = GeneticAlgorithm(population_size=args.ga_population_size, env_params=ENV_PARAMS)

    print("Initialized GA")

    gpu_num = None
    if args.agent_gpu > -1:
        gpu_num = args.agent_gpu
    test_agents = [agent.DQNAgent(i, ENV_PARAMS, net_filepath=args.agent_initial_weights, gpu_num=gpu_num) for i in range(args.num_agents)]

    if not os.path.isdir('Pickled/Final/KavyaRuns/Coevolution/'):
        os.mkdir('Pickled/Final/KavyaRuns/Coevolution/')
    run_folder = f'Pickled/Final/KavyaRuns/Coevolution/Coevolution{args.num_agents}DQNAgent_{args.ga_tile_size}tile_{args.ga_rate_elitism}elitism_{args.ga_rate_mutation}mutation_{args.num_food}food_{args.num_blockade}blocks/'
    if not os.path.isdir(run_folder):
        os.mkdir(run_folder)
    duel_train_filename = f'{run_folder}Coevolution{args.num_agents}DQNAgent_{args.ga_tile_size}tile_{args.ga_rate_elitism}elitism_{args.ga_rate_mutation}mutation_{args.num_food}food_{args.num_blockade}blocks'
    temp_ga_filename = f'{run_folder}tempGAdata'
    print("Initialized agents")

    duel_training_pickle = {}

    pickle_dict = None
    grids, fitness_values = None, None
    with open(args.ga_initial_population, 'rb') as f:
        pickle_dict = pickle.load(f)

    for iteration in range(args.duel_train_iterations):
        print("iteration ", iteration)
        grids, fitness_values = pickle_dict['grids'], pickle_dict['fitness values']

        with open(temp_ga_filename, 'wb') as f:
            pickle.dump(pickle_dict, f)

        elitism_size = int(args.ga_population_size*args.ga_rate_elitism)
        selected = sorted(range(len(fitness_values[-1])), key=lambda x: fitness_values[-1][x], reverse=True)[:elitism_size]
        elitism_population = [grids[-1][x] for x in range(len(fitness_values[-1])) if x in selected]

        episode_rewards, episode_loss = train_dqn.dqn_main(ENV_PARAMS, test_agents, 
                                                            grids = elitism_population, 
                                                            random_proportion=0.2,
                                                            filename=f'{run_folder}target_net_{iteration}iteration.pt', 
                                                            num_episodes=10, verbose=True)
        print("\ttrained DQN")
        grids, fitness_values = ga.run(rate_elitism=args.ga_rate_elitism, 
                                        rate_mutation=args.ga_rate_mutation, 
                                        iterations=args.ga_iterations, 
                                        agents=test_agents, 
                                        verbose=True, tqdm_disable=False, tile_size=args.ga_tile_size,
                                        filename=temp_ga_filename) 
        pickle_dict = {
            'train_episode_rewards': episode_rewards, 
            'train_episode_loss': episode_loss,
            'grids': grids,
            'fitness values': fitness_values,
            'env_params': ENV_PARAMS
        }

        duel_training_pickle[iteration] = pickle_dict
        with open(duel_train_filename, 'wb') as f:
            pickle.dump(duel_training_pickle, f)

    print(duel_train_filename)