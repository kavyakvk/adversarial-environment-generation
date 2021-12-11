import random
import environment
import utils
import numpy as np
from tqdm import tqdm
import copy
import pickle

class GeneticAlgorithm:
    def get_fitness(self, grid, agents):
        env = environment.Environment(self.env_params, grid)
        for agent in agents:
            agent.set_spt(env.spt)
        food_collected = env.run_episode(agents, visualize=False)
        return -1*food_collected/len(agents)
    
    def fix_food(self, grid):
        total_food = utils.calculate_food(grid, self.env_params)
        if total_food < self.env_params['grid']['food']:
            while total_food != self.env_params['grid']['food']: 
                # if the amount of placed food is too low
                food_idxs = list(zip(*np.where(grid >= self.env_params['coding_dict']['food_start'])))
                empty_idxs = list(zip(*np.where(grid == self.env_params['coding_dict']['empty'])))
                food_idxs.extend(empty_idxs)
                random.shuffle(food_idxs)
                for idx in food_idxs:
                    x,y = idx[0], idx[1]
                    if total_food == self.env_params['grid']['food']:
                        break
                    else:
                        cell_food = 0
                        if grid[x,y] != self.env_params['coding_dict']['empty']:
                            cell_food = utils.get_food(grid[x,y], self.env_params)
                        if cell_food < self.env_params['max_food']:
                            delta_food = self.env_params['grid']['food']-(total_food-cell_food)
                            if delta_food > 0:
                                food = random.randint(cell_food+1, min(self.env_params['max_food'], delta_food))
                            total_food = total_food+(food-cell_food)
                            grid[x,y] = utils.get_grid_food(food, self.env_params)
                    #total_food = utils.calculate_food(grid, self.env_params) 
        elif total_food > self.env_params['grid']['food']:
            # if the amount of placed food is too high
            while total_food != self.env_params['grid']['food']:
                food_idxs = list(zip(*np.where(grid >= self.env_params['coding_dict']['food_start'])))
                random.shuffle(food_idxs)
                for idx in food_idxs:
                    x,y = idx[0], idx[1]
                    if total_food == self.env_params['grid']['food']:
                        break
                    else:
                        cell_food = utils.get_food(grid[x,y], self.env_params)
                        delta_food = self.env_params['grid']['food']-(total_food-cell_food)
                        food = delta_food
                        if delta_food < 0:
                            food = random.randint(max(0, cell_food+delta_food), cell_food-1)
                        total_food = total_food+(food-cell_food)
                        if food == 0:
                            grid[x,y] = self.env_params['coding_dict']['empty']
                        else:
                            grid[x,y] = utils.get_grid_food(food, self.env_params)
                    #total_food = utils.calculate_food(grid, self.env_params) 
        return grid

    def fix_blockade(self, grid):
        total_blockades = utils.calculate_blockades(grid, self.env_params)
        if total_blockades > self.env_params['grid']['blockade']:
            blockades_idxs = list(zip(*np.where(grid == self.env_params['coding_dict']['blockade'])))
            random.shuffle(blockades_idxs)
            for idx in range(total_blockades - self.env_params['grid']['blockade']):
                x, y = blockades_idxs[idx][0], blockades_idxs[idx][1]
                grid[x, y] = self.env_params['coding_dict']['empty']
        return grid

    def get_crossover(self, grid1, grid2, tile_size=2):
        assert(self.env_params['N']%tile_size == 0)
        assert(self.env_params['M']%tile_size == 0)

        n = self.env_params['N']//tile_size
        m = self.env_params['M']//tile_size
        mask1 = np.kron(np.random.randint(low=0, high=2, size=(n,m)), np.ones((tile_size,tile_size)))
        mask2 = np.ones((self.env_params['N'], self.env_params['M'])) - mask1
        crossed = [np.multiply(mask1, grid1) + np.multiply(mask2, grid2)]
        crossed.append(np.multiply(mask2, grid1) + np.multiply(mask1, grid2))

        return crossed
    
    def get_mutated(self, original_grid, rate_mutation):
        grid = copy.deepcopy(original_grid)
        total_food = self.env_params['grid']['food']
        total_blockades = len(np.where(grid == self.env_params['coding_dict']['blockade'])[0])
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if random.random() < rate_mutation:
                    if grid[x,y] >= self.env_params['coding_dict']['food_start']:
                        # randomly add / subtract food
                        food = random.randint(0, self.env_params['max_food'])
                        total_food = total_food-utils.get_food(grid[x,y], self.env_params)+food
                        if food == 0:
                            grid[x,y] = self.env_params['coding_dict']['empty']
                        else:
                            grid[x,y] = utils.get_grid_food(food, self.env_params)
                    elif grid[x,y] == self.env_params['coding_dict']['blockade']:
                        # flip blockade to empty
                        grid[x,y] = self.env_params['coding_dict']['empty']
                        total_blockades -= 1
                    elif grid[x,y] == self.env_params['coding_dict']['empty']:
                        if random.random() >= 0.5:
                            # flip empty to food
                            food = random.randint(1, self.env_params['max_food'])
                            total_food = total_food+food
                            grid[x,y] = utils.get_grid_food(food, self.env_params)
                        else:
                            # flip empty to blockade
                            grid[x,y] = self.env_params['coding_dict']['blockade']
                            total_blockades += 1
        return grid

    def check_feasibility(self, grid):
        spt = [[q[0] for q in r] for r in utils.expert_navigation_policy_set(grid, (0,0), self.env_params)]
        for x in range(self.env_params['N']):
            for y in range(self.env_params['M']):
                if len(spt[x][y]) == 0 and grid[x][y] >= self.env_params['coding_dict']['food_start']:
                    return False
        return True

    def generate_n_valid_feasible_grids(self, n):

        population = [utils.generate_random_grid(self.env_params) for i in range(n)]

        for i in range(len(population)):
            grid = population[i]
            # check feasibility of solution
            while not self.check_feasibility(grid):
                population[i] = utils.generate_random_grid(self.env_params)
                grid = population[i]
        return population

    def __init__(self, population_size, env_params):
        self.population_size = population_size
        self.env_params = env_params
        self.population = self.generate_n_valid_feasible_grids(self.population_size)
    
    def run(self, rate_elitism, rate_mutation, iterations, agents, verbose=False, tqdm_disable=True, tile_size=2, filename=None, continue_training = False):
        grids, fitness_values = [], []
        # continue training from results in a previously pickled file
        if continue_training:
            assert filename is not None
            pickle_file = open(filename, "rb")
            pickle_dict = pickle.load(pickle_file)
            grids, fitness_values = pickle_dict['grids'], pickle_dict['fitness values']
            pickle_file.close()
            self.population = copy.deepcopy(grids[-1])
            
        for i in range(iterations):
            fitness = [self.get_fitness(x, agents) for x in tqdm(self.population, disable=tqdm_disable, position=0, leave=True)]
            if verbose:
                print("ITERATION ", i, sum(fitness)/len(fitness))
            
            # store the grids and fitness values
            grids.append(self.population)
            fitness_values.append(fitness)

            # elitism based selection
            num_selected = int(self.population_size*rate_elitism)
            selected = sorted(range(len(fitness)), key=lambda x: fitness[x], reverse=True)[:num_selected]
            selected_population = [self.population[x] for x in range(len(fitness)) if x in selected]

            # crossover
            num_crossover = (self.population_size - num_selected)//2
            new_population = []
            for j in range(num_crossover):
                crossover_idxs = random.sample(range(0, self.population_size), 2)
                crossed = self.get_crossover(self.population[crossover_idxs[0]], 
                                             self.population[crossover_idxs[1]], tile_size)
                new_population.extend(crossed)
            
            # mutation
            mutation_candidates = random.sample([x for x in range(len(new_population))], num_crossover)
            for j in range(len(mutation_candidates)):
                grid = self.get_mutated(new_population[j], rate_mutation)
                grid = self.fix_blockade(self.fix_food(grid))
                utils.check_valid(grid, self.env_params)
                # check feasibility of solution
                if self.check_feasibility(grid):
                    new_population[j] = grid
                else:
                    j-=1 

            feasible_new_population = []
            for j in range(len(new_population)):
                grid = new_population[j]
                # Non-mutated may need fixing
                if not utils.check_valid(grid, self.env_params, throw_error=False):
                    grid = self.fix_blockade(self.fix_food(grid))
                # Non-mutated need a feasibility check
                if self.check_feasibility(grid):
                    feasible_new_population.append(grid)
            
            num_missing = len(new_population)-len(feasible_new_population)
            feasible_new_population.extend(self.generate_n_valid_feasible_grids(num_missing))
                    
            #update population
            selected_population.extend(feasible_new_population)
            self.population = selected_population
            assert(len(self.population) == self.population_size)

            #optionally pickle files
            if filename:
                pickle_dict = {
                    'grids': grids,
                    'fitness values': fitness_values,
                    'env_params': self.env_params
                }
                with open(filename, 'wb') as f:
                    pickle.dump(pickle_dict, f)
        return grids, fitness_values
        


