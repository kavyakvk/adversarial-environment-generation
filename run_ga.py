from geneticalg import GeneticAlgorithm
import agent
import copy
import pickle
import argparse

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
    parser.add_argument('-a', '--agent_type', action='append', choices=['DQN', 'Random', 'Swarm'], required=True)
    parser.add_argument('--agent_gpu', default=-1, type=int)

    parser.add_argument('--ga_population_size', default=100, type=int)
    parser.add_argument('--ga_rate_elitism', default=0.2, type=float)
    parser.add_argument('--ga_rate_mutation', default=0.1, type=float)
    parser.add_argument('--ga_iterations', default=50, type=float)
    parser.add_argument('--ga_tile_size', default=2, type=int)

    args = parser.parse_args()

    # Override global environment parameters
    ENV_PARAMS['grid']['food'] = args.num_food
    ENV_PARAMS['grid']['blockade'] = args.num_blockade

    ga = GeneticAlgorithm(population_size=args.ga_population_size, env_params=ENV_PARAMS)

    print("Initialized GA")

    test_agents = None
    if 'DQN' in args.agent_type:
        gpu_num = None
        if args.agent_gpu > -1:
            gpu_num = args.agent_gpu
        test_agents = [agent.DQNAgent(i, ENV_PARAMS, net_filepath="DQN/target_net.pt", gpu_num=gpu_num) for i in range(args.num_agents)]
    elif 'Random' in args.agent_type:
        test_agents = [agent.RandomAgent(i, ENV_PARAMS) for i in range(args.num_agents)]
    elif 'Swarm' in args.agent_type:
        test_agents = [agent.SwarmAgent(i, ENV_PARAMS) for i in range(args.num_agents)]

    print("Initialized agents")

    filename = f'Pickled/Final/KavyaRuns/GA_{args.num_agents}{args.agent_type[0]}Agent_{args.ga_tile_size}tile_{args.ga_rate_mutation}mutation_{args.num_food}food_{args.num_blockade}blocks'
    grids, fitness_values = ga.run(rate_elitism=args.ga_rate_elitism, 
                                    rate_mutation=args.ga_rate_mutation, 
                                    iterations=args.ga_iterations, 
                                    agents=test_agents, 
                                    verbose=True, tqdm_disable=False, tile_size=args.ga_tile_size,
                                    filename=filename) 
    print(filename)