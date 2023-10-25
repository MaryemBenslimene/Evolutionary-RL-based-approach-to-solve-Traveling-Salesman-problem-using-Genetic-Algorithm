import torch
import torch.optim as optim
import random
from model import DRL4TSP
from tasks import tsp
import copy
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAOptimizer(optim.Optimizer):

    def __init__(self, generations, mutation_rate, population_size, elite_pct):
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.elite_pct = elite_pct
        self.population = None
        

    def create_individual(self, statistic_size, dynamic_size, update_fn, args):
        return DRL4TSP(statistic_size, dynamic_size, 
                              args.hidden_size, update_fn, 
                              tsp.update_mask, 
                              args.num_layers, 
                              args.dropout).to(device)
    
    def create_population(self, statistic_size, dynamic_size, update_fn, args):
        return [self.create_individual(statistic_size, dynamic_size, update_fn, args) for indiv in range(self.population_size)]
        
    def fitness(self, reward, fitness):
        fitness.append(reward)
    
    def parents_selection(self, fitness):
        parents = []
        for _ in range(self.population_size//2):
            parent_index = fitness.index(min(fitness))
            parents.append(self.population[parent_index])
            fitness.pop(parent_index)
            self.population.pop(parent_index)

        print("parents len :", len(parents))

        return parents


    def crossover(self, parents, statistic_size, dynamic_size, update_fn, args):
        num_parents = len(parents)
        num_pairs = num_parents // 2
        offspring = []
        for i in range(num_pairs):
            parent1 = parents[i*2]
            parent2 = parents[i*2+1]

            parent1_weights = torch.cat([param.view(-1) for param in parent1.parameters()])
            parent2_weights = torch.cat([param.view(-1) for param in parent2.parameters()])

            child1 = self.create_individual(statistic_size, dynamic_size, update_fn, args)
            child2 = self.create_individual(statistic_size, dynamic_size, update_fn, args)

            crossover_point = np.random.randint(1, len(parent1_weights)-1)
            child1_weights = torch.cat([parent1_weights[:crossover_point], parent2_weights[crossover_point:]])
            child1_state_dict = {}
            idx = 0
            for name, param in child1.named_parameters():
                shape = param.shape
                num_params = torch.prod(torch.tensor(shape))
                child1_state_dict[name] = child1_weights[idx:idx+num_params].view(shape)
                idx += num_params
            child1.load_state_dict(child1_state_dict)
            offspring.append(child1)

            child2_weights = torch.cat([parent2_weights[:crossover_point], parent1_weights[crossover_point:]])
            child2_state_dict = {}
            idx = 0
            for name, param in child2.named_parameters():
                shape = param.shape
                num_params = torch.prod(torch.tensor(shape))
                child2_state_dict[name] = child2_weights[idx:idx+num_params].view(shape)
                idx += num_params
            child2.load_state_dict(child2_state_dict)
            offspring.append(child2)

        # If the number of parents is odd, add one more offspring
        if num_parents % 2 == 1:
            parent1 = parents[-1]
            parent2 = parents[-2]
            parent1_weights = torch.cat([param.view(-1) for param in parent1.parameters()])
            parent2_weights = torch.cat([param.view(-1) for param in parent2.parameters()])
            child = self.create_individual(statistic_size, dynamic_size, update_fn, args)
            child_weights = torch.cat([parent1_weights[:crossover_point], parent2_weights[crossover_point:]])
            child_state_dict = {}
            idx = 0
            for name, param in child.named_parameters():
                shape = param.shape
                num_params = torch.prod(torch.tensor(shape))
                child_state_dict[name] = child_weights[idx:idx+num_params].view(shape)
                idx += num_params
            child.load_state_dict(child_state_dict)
            offspring.append(child)
        print("len off", len(offspring))

        return offspring  # Keep only the first num_parents offspring


    def mutation(self):
        mutated_population = []
        for individual in self.population:
            mutated_individual = copy.deepcopy(individual)
            for name, param in mutated_individual.named_parameters():
                if random.random() < abs(param).sum() / param.numel() * self.mutation_rate:
                    print("Mutation")
                    mutated_value = torch.normal(0, 1, size=param.shape).to(param.device)
                    new_param = param + mutated_value
                    mutated_individual.state_dict()[name].copy_(new_param)
            mutated_population.append(mutated_individual)
        self.population = mutated_population
        print("pop len", len(self.population))


    """  
    def parents_selection(self, fitness):
        sorted_indices = sorted(range(len(fitness)), key=lambda k: fitness[k])
        rank = [0] * len(fitness)
        for i in range(len(fitness)):
            rank[sorted_indices[i]] = i + 1
        total_rank = sum(rank)
        probabilities = [r / total_rank for r in rank]
        parents = []
        for i in range(self.population_size // 2):
            parent1_index = np.random.choice(len(fitness), p=probabilities)
            parent1 = self.population[parent1_index]
            fitness.pop(parent1_index)
            self.population.pop(parent1_index)
            rank.pop(parent1_index)
            probabilities = [r / sum(rank) for r in rank]
            parent2_index = np.random.choice(len(fitness), p=probabilities)
            parent2 = self.population[parent2_index]
            fitness.pop(parent2_index)
            self.population.pop(parent2_index)
            rank.pop(parent2_index)
            parents.extend([parent1, parent2])

        print("lrn parents", len(parents))
        return parents

    def parents_selection(self, fitness):
        parents = []
        for _ in range(self.population_size//4):
            parent_index = fitness.index(min(fitness))
            parents.append(self.population[parent_index])
            fitness.pop(parent_index)
            self.population.pop(parent_index)

        print("parents : ", parents)
        print("parents len :", len(parents))
        return parents"""

'''''''''
    def best_model(self, data_loader, reward_fn, statistic_size, dynamic_size, update_fn, args):
        self.population = self.create_population(statistic_size, dynamic_size, update_fn, args)
        for generation in range(self.generations):
            fitness = self.fitness(data_loader, reward_fn)
            parents = self.parents_selection(fitness)
            offsprings = self.crossover(parents)
            self.population = parents + offsprings
            self.population = self.mutation()
        best_model_index = fitness.index(max(fitness))
        return self.population[best_model_index]
'''''''''

''''''''''
parser = argparse.ArgumentParser(description='Combinatorial Optimization')
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--task', default='vrp')
parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
parser.add_argument('--actor_lr', default=5e-4, type=float)
parser.add_argument('--critic_lr', default=5e-4, type=float)
parser.add_argument('--max_grad_norm', default=2., type=float)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--layers', dest='num_layers', default=1, type=int)
parser.add_argument('--train-size',default=1000000, type=int)
parser.add_argument('--valid-size', default=1000, type=int)

from tasks import vrp
from tasks.vrp import VehicleRoutingDataset
from torch.utils.data import DataLoader

args = parser.parse_args()
LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
MAX_DEMAND = 9
max_load = LOAD_DICT[args.num_nodes]

train_data = VehicleRoutingDataset(args.train_size,
                                    args.num_nodes,
                                    max_load,
                                    MAX_DEMAND,
                                    args.seed)

test_data = VehicleRoutingDataset(args.valid_size,
                                  args.num_nodes,
                                  max_load,
                                  MAX_DEMAND,
                                  args.seed + 2)

test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)

update_fn = None

GAoptim = GAOptimizer(generations=50, mutation_rate=0.1, population_size=100)
actor = GAoptim.best_model(test_loader, vrp.reward, statistic_size = 2, dynamic_size = 1, update_fn=update_fn, args=args)

print("actor = ", actor.get_weights())

'''''''''''