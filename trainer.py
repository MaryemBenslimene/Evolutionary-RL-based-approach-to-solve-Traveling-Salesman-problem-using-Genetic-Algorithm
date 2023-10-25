"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""
import wandb
import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torch.utils.data import DataLoader
from itertools import product

import submitit
from datetime import datetime
from pathlib import Path

from model import DRL4TSP, Encoder
import GA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_reward = np.inf


    actor.train()
    critic.train()

    times, losses, rewards, critic_rewards = [], [], [], []

    epoch_start = time.time()
    start = epoch_start

    for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                #print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                #      (batch_idx, len(train_loader), mean_reward, mean_loss,
                #       times[-1]))

    mean_loss = np.mean(losses)
    mean_reward = np.mean(rewards)

        # Save the weights
    epoch_dir = os.path.join(checkpoint_dir, '%s' )
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)

    save_path = os.path.join(epoch_dir, 'actor.pt')
    torch.save(actor.state_dict(), save_path)

    save_path = os.path.join(epoch_dir, 'critic.pt')
    torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
    valid_dir = os.path.join(save_dir, '%s')

    mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)

        # Save best model parameters
    if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

    print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
            '(%2.4fs / 100 batches)\n' % \
            (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
            np.mean(times)))



def train_tsp(args):

    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44

    from tasks import tsp
    from tasks.tsp import TSPDataset


    #wandb.init(project="PFA-routing" + str(config['generations']), config=config, mode='online')       
    #wandb.run.name = wandb.run.name = "Generations" + str(config['generations'] )+'mutation_rate' + str(config['mutation_rate']) + 'population_size' + str(config['population_size']) +'elite_pct'+ str(config['elite_pct'])
    #wandb.config.update(config)

    wandb.init(project = 'PFA tests' , mode = 'online')
    wandb.run.name = 'test PFA with 20 nodes'

    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 1 # dummy for compatibility

    train_data = TSPDataset(args.num_nodes, args.train_size)
    valid_data = TSPDataset(args.num_nodes, args.valid_size)

    update_fn = None

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    #GAOptim = GA.GAOptimizer(generations=config['generations'], mutation_rate=config['mutation_rate'], population_size=config['population_size'], elite_pct=config['population_size'])
    
    GAOptim = GA.GAOptimizer(generations=100, mutation_rate=0.1, population_size=20, elite_pct=0.1)
    GAOptim.population = GAOptim.create_population(STATIC_SIZE, DYNAMIC_SIZE, update_fn, args)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tsp.reward
    kwargs['render_fn'] = tsp.render

    for generation in range(GAOptim.generations):
        print("started")
        print("gen", generation)
        fitness = []    
        wandb.log({'Generation': generation})
        for individual in GAOptim.population :
            print('indiv')
            if args.checkpoint:
                path = os.path.join(args.checkpoint, 'actor.pt')
                individual.load_state_dict(torch.load(path, device))

                path = os.path.join(args.checkpoint, 'critic.pt')
                critic.load_state_dict(torch.load(path, device))

            if not args.test:
                train(individual, critic, **kwargs)

            test_data = TSPDataset(args.num_nodes, args.train_size)
            test_dir = 'test'
            test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
            reward = validate(test_loader, individual, tsp.reward, tsp.render, test_dir, num_plot=5)

            fitness.append(reward)
            wandb.log({'Validation reward': reward})

        mean = sum(fitness) / GAOptim.population_size
        wandb.log({'Mean fitness over generations': mean})
        parents = GAOptim.parents_selection(fitness)
        offsprings = GAOptim.crossover(parents, STATIC_SIZE, DYNAMIC_SIZE, update_fn, args)
        GAOptim.population = parents + offsprings
        GAOptim.mutation()
    best_model_index = fitness.index(min(fitness))
    actor = GAOptim.population[best_model_index]
    print('Average tour length: ', fitness[best_model_index])

    wandb.finish()




def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed)

    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1)

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    print('Average tour length: ', out)

'''''
PARAM_GRID = list(product(
    [10,20],  # generations
    [0.1,0.2,0.3],  # mutation_rate
    [10],  # population_size
    [0.1],  # elite_pct
))
'''''



#parser.add_argument('--seed', default=12345, type=int)


#parser.add_argument('--generations', type=int, default=10)
#parser.add_argument('--mutation_rate', type=float, default=0.1)
#parser.add_argument('--population_size', type=int, default=5)
#parser.add_argument('--elite_pct', type=float, default=0.2)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='tsp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=6., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=10000, type=int) # na9ast zouz sfar
    parser.add_argument('--valid-size', default=1000, type=int)


    args = parser.parse_args()
    '''''
    hyper_list = []
    for param_ix in range(len(PARAM_GRID)):

        params = PARAM_GRID[param_ix]

        g,m,p,e = params
        if len(sys.argv) > 1:
            g = args.generations
            m = args.mutation_rate
            p = args.population_size
            e = args.elite_pct
        config = {}
        config['generations'] = g
        config['mutation_rate'] = m
        config['population_size']=p
        config['elite_pct'] = e
        if config not in hyper_list:
            hyper_list += [config]
    '''''

    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)

    d = datetime.today()
    exp_dir = (
            Path("./submitit-res")
            / f"{d.strftime('%Y-%m-%d')}_20nodesrand_eval"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    submitit_logdir = exp_dir / "submitit_logs"
    executor = submitit.AutoExecutor(folder=submitit_logdir)
    executor.update_parameters(
        timeout_min= 60 * 300,
        #slurm_partition="long",
        slurm_additional_parameters={"account": "def-bengioy"},
        gpus_per_node=1,
        # tasks_per_node=1,
        # cpus_per_task=10,
        slurm_mem="16G"
    )

    if args.task == 'tsp':
        #job = executor.map_array(train_tsp,hyper_list)
        job = executor.submit(train_tsp,args)
    elif args.task == 'vrp':
        train_vrp(args)
    else:
        raise ValueError('Task <%s> not understood'%args.task)
    
    print(f"# Submitted job")
    #print(f"# Submitted job {len(hyper_list)}")

    

    
