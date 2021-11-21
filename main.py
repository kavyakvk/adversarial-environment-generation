from geneticalg import GeneticAlgorithm
from agent import RandomAgent
import copy

# ant black, obstacle blue, hive yellow/brown, food green, empty 

ENV_PARAMS = {'coding_dict': {'empty': 0, 'agent': 1, 'bounds': 2, 'hive': 3, 'blockade': 4, 
                                'food_start': 5},
                'N': 20, 'M': 20, 'max_food': 5, 'observation_radius': 5, 'steps': 5000, 'spawn_rate': 2,
                'pheromone': {'evaporation': 0.1, 'diffusion': 0.1, 'step': 0.2, 'cap': 5},
                'grid': {'food': 40, 'blockade': 20}}

rgb_coding = {ENV_PARAMS['coding_dict']['empty']: [0, 0, 0], #white
                ENV_PARAMS['coding_dict']['agent']: [150, 0, 150], #purple
                ENV_PARAMS['coding_dict']['bounds']: [100,100,100], #grey
                ENV_PARAMS['coding_dict']['hive']: [150,150,0], #yellow
                ENV_PARAMS['coding_dict']['blockade']: [45,0,255], #blue
                ENV_PARAMS['coding_dict']['food_start']: [0,255,45]}
for food in range(1,10):
    color = copy.deepcopy(rgb_coding[ENV_PARAMS['coding_dict']['food_start']])
    color[1] -= food*5
    color[2] += food*5
    rgb_coding[ENV_PARAMS['coding_dict']['food_start']+food] = tuple(color)
ENV_PARAMS['rgb_coding'] = rgb_coding

print(ENV_PARAMS)

ga = GeneticAlgorithm(100, ENV_PARAMS)
ga.run(rate_elitism=0.1, rate_mutation=0.1, iterations=100, agents=[None, None, None])