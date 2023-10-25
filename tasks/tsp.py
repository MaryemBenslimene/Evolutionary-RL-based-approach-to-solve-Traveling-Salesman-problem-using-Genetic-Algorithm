"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None):
        super(TSPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset = torch.rand((num_samples, 2, size))
        self.dynamic = torch.zeros(num_samples, 1, size)
        self.num_nodes = size
        self.size = num_samples
        

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, tour_indices):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()


def update_dynamic(self, dynamic, chosen_idx):
    """Updates the (load, demand) dataset values."""

    # Update the dynamic elements differently for if we visit depot vs. a city
    visit = chosen_idx.ne(0)
    depot = chosen_idx.eq(0)

    # Clone the dynamic variable so we don't mess up graph
    all_loads = dynamic[:, 0].clone()
    all_demands = dynamic[:, 1].clone()

    load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
    demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

    # Across the minibatch - if we've chosen to visit a city, try to satisfy
    # as much demand as possible
    if visit.any():

        new_load = torch.clamp(load - demand, min=0)
        new_demand = torch.clamp(demand - load, min=0)

        # Broadcast the load to all nodes, but update demand seperately
        visit_idx = visit.nonzero().squeeze()

        all_loads[visit_idx] = new_load[visit_idx]
        all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
        all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

    # Return to depot to fill vehicle load
    if depot.any():
        all_loads[depot.nonzero().squeeze()] = 1.
        all_demands[depot.nonzero().squeeze(), 0] = 0.

    tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
    return torch.tensor(tensor.data, device=dynamic.device)

def render(static, tour_indices, save_path):
    """Plots the found tours."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
