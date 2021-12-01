from geneticalg import GeneticAlgorithm
import agent
import copy
import pickle

# ant black, obstacle blue, hive yellow/brown, food green, empty 

<<<<<<< HEAD
ENV_PARAMS = {'coding_dict': {'empty': 0, 'agent': 1, 'bounds': 2, 'hive': 3, 'blockade': 4, 'food_start': 5}, 
                            'N': 20, 'M': 20, 'max_food': 5, 'observation_radius': 3, 'steps': 300, 'spawn_rate': 2, 
=======
ENV_PARAMS = {'coding_dict': {'empty': 0, 'agent': 1, 'bounds': 2, 'hive': 3, 'blockade': 4, 'food_start': 6}, 
                            'N': 10, 'M': 10, 'max_food': 5, 'observation_radius': 1, 'steps': 300, 'spawn_rate': 2, 
>>>>>>> 5dc68310ab68e13b85b1db0ed40caf3a18187808
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

ga = GeneticAlgorithm(population_size=100, env_params=ENV_PARAMS)
test_agents = [agent.SwarmAgent(i, ENV_PARAMS) for i in range(5)]
grids, fitness_values = ga.run(rate_elitism=0.1, rate_mutation=0.1, iterations=100, agents=test_agents, verbose=True, tdqm_disable=False) 
print('grids:', grids)
print('fitness values:', fitness_values)
pickle_dict = {
    'grids': grids,
    'fitness values': fitness_values,
    'env_params': ENV_PARAMS
}
with open('Pickled/GA_SwarmAgent_5agents_0.1elitism_0.1mutation_40food_20blocks', 'wb') as f:
    pickle.dump(pickle_dict, f)