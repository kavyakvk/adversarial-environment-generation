import random
import environment
import utils
import numpy as np
from tqdm import tqdm
import copy

class GeneticAlgorithm:
    def calculate_food(self, grid):
        counts = np.bincount(grid.flatten().astype('int64'))
        food_counts = counts[self.env_params['coding_dict']['food_start']:]
        assert(len(food_counts) <= self.env_params['max_food'])
        if len(food_counts) != self.env_params['max_food']:
            food_counts = np.append(food_counts, np.zeros(self.env_params['max_food']-len(food_counts)))
        total_food = np.dot(np.arange(1, self.env_params['max_food']+1), food_counts)
        return total_food

    def calculate_blockades(self, grid):
        counts = np.bincount(grid.flatten().astype('int64'))
        return counts[self.env_params['coding_dict']['blockade']]

    def check_valid(self, grid, throw_error=True):
        counts = np.bincount(grid.flatten().astype('int64'))
        num_blockades = self.calculate_blockades(grid)
        total_food = self.calculate_food(grid)
        if throw_error:
            assert(num_blockades <= self.env_params['grid']['blockade'])
            assert(total_food == self.env_params['grid']['food'])
        else:
            if ((num_blockades <= self.env_params['grid']['blockade']) 
                    and (total_food == self.env_params['grid']['food'])):
                return True
            return False

    def get_food(self, cell_value):
        # helper method transforming cell value to amount of food
        assert(cell_value >= self.env_params['coding_dict']['food_start'])
        food = cell_value-self.env_params['coding_dict']['food_start']+1
        assert(food <= self.env_params['max_food'])
        return food
    
    def get_baseline_food(self, cell_value, total_food):
        # helper method to calculate total food minus the current cell's food
        total_food = total_food-self.get_food(cell_value)
    
    def get_grid_food(self, food):
        # helper method transforming amount of food to cell value
        assert(food <= self.env_params['max_food'])
        return self.env_params['coding_dict']['food_start']+food-1

    def generate_random_grid(self):
        # randomly select the number of blockades
        num_blockades = random.randint(1, self.env_params['grid']['blockade'])

        grid = np.zeros(self.env_params['N']*self.env_params['M']-num_blockades)

        # insert the blockades into the grid
        grid = np.insert(grid, np.random.choice(len(grid), size=num_blockades), np.ones(num_blockades))
        grid = grid*self.env_params['coding_dict']['blockade']

        grid[0] = self.env_params['coding_dict']['hive']

        # place food across the grid
        food_left = self.env_params['grid']['food']
        potential_food_idxs = list(np.where(grid == 0)[0])
        random.shuffle(potential_food_idxs)
        while food_left > 0:
            food = random.randint(1, min(food_left, self.env_params['max_food']))
            food_left -= food
            grid[potential_food_idxs[0]] = self.get_grid_food(food)
            del potential_food_idxs[0]

        grid = np.reshape(grid, (self.env_params['N'],self.env_params['M']))

        self.check_valid(grid)
        return grid

    def get_fitness(self, grid, agents):
        env = environment.Environment(self.env_params, grid)
        for agent in agents:
            agent.set_spt(env.spt)
        food_collected = env.run_episode(agents, visualize=False)
        return -1*food_collected/len(agents)
    
    def fix_food(self, grid):
        total_food = self.calculate_food(grid)
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
                            cell_food = self.get_food(grid[x,y])
                        if cell_food < self.env_params['max_food']:
                            delta_food = self.env_params['grid']['food']-(total_food-cell_food)
                            if delta_food > 0:
                                food = random.randint(cell_food+1, min(self.env_params['max_food'], delta_food))
                            total_food = total_food+(food-cell_food)
                            grid[x,y] = self.get_grid_food(food)
                    #total_food = self.calculate_food(grid) 
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
                        cell_food = self.get_food(grid[x,y])
                        delta_food = self.env_params['grid']['food']-(total_food-cell_food)
                        food = delta_food
                        if delta_food < 0:
                            food = random.randint(max(0, cell_food+delta_food), cell_food-1)
                        total_food = total_food+(food-cell_food)
                        if food == 0:
                            grid[x,y] = self.env_params['coding_dict']['empty']
                        else:
                            grid[x,y] = self.get_grid_food(food)
                    #total_food = self.calculate_food(grid) 
        return grid

    def fix_blockade(self, grid):
        total_blockades = self.calculate_blockades(grid)
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
                        total_food = total_food-self.get_food(grid[x,y])+food
                        if food == 0:
                            grid[x,y] = self.env_params['coding_dict']['empty']
                        else:
                            grid[x,y] = self.get_grid_food(food)
                    elif grid[x,y] == self.env_params['coding_dict']['blockade']:
                        # flip blockade to empty
                        grid[x,y] = self.env_params['coding_dict']['empty']
                        total_blockades -= 1
                    elif grid[x,y] == self.env_params['coding_dict']['empty']:
                        if random.random() >= 0.5:
                            # flip empty to food
                            food = random.randint(1, self.env_params['max_food'])
                            total_food = total_food+food
                            grid[x,y] = self.get_grid_food(food)
                        else:
                            # flip empty to blockade
                            grid[x,y] = self.env_params['coding_dict']['blockade']
                            total_blockades += 1
        return grid

    def check_feasibility(self, grid):
        spt = [[q[0] for q in r] for r in utils.expert_navigation_policy_set(grid, (0,0))]
        for x in range(self.env_params['N']):
            for y in range(self.env_params['M']):
                if len(spt[x][y]) == 0 and grid[x][y] >= self.env_params['coding_dict']['food_start']:
                    return False
        return True

    def __init__(self, population_size, env_params):
        self.population_size = population_size
        self.env_params = env_params
        self.population = [self.generate_random_grid() for i in range(self.population_size)]
    
    def run(self, rate_elitism, rate_mutation, iterations, agents, verbose=False):
        tdqm_disable = not verbose
        grids, fitness_values = [], []
        for i in range(iterations):
            fitness = [self.get_fitness(x, agents) for x in tqdm(self.population, disable=tdqm_disable)]
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
                                             self.population[crossover_idxs[1]])
                new_population.extend(crossed)
            
            # mutation
            mutation_candidates = random.sample([x for x in range(len(new_population))], num_crossover)
            for j in range(len(mutation_candidates)):
                grid = self.get_mutated(new_population[j], rate_mutation)
                grid = self.fix_blockade(self.fix_food(grid))
                self.check_valid(grid)
                # check feasibility of solution
                if self.check_feasibility(grid):
                    new_population[j] = grid
                else:
                    j-=1 

            for grid in new_population:
                # Non-mutated may need fixing
                if not self.check_valid(grid, throw_error=False):
                    grid = self.fix_blockade(self.fix_food(grid))
            
            #update population
            selected_population.extend(new_population)
            self.population = selected_population
            assert(len(self.population) == self.population_size)
        return grids, fitness_values
        



