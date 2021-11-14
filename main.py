from geneticalg import GeneticAlgorithm
from agent import RandomAgent

ENV_PARAMS = {'coding_dict': {'empty': 0, 'agent': 1, 'bounds': 2, 'hive': 3, 'blockade': 4, 
                                'food_start': 5},
                'N': 20, 'M': 20, 'max_food': 5, 'observation_radius': 4, 'steps': 5000, 'spawn_rate': 2,
                'pheromone': {'evaporation': 0.1, 'diffusion': 0.1, 'step': 0.2},
                'grid': {'food': 40, 'blockade': 20}}

ga = GeneticAlgorithm(10, ENV_PARAMS)
ga.run(0.1, 0.1, 10, [None, None, None])