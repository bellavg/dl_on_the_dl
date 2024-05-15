import numpy as np
import os
import torch


class NBodyDataset:
    def __init__(self, dataroot, phase, suffix='_charged5_initvel1small', max_samples=1000):
        self.suffix = suffix  # '_charged5_initvel1small'
        self.data_root = dataroot  # 'nbody_dataset'
        self.max_samples = int(max_samples)
        self.phase = phase  # train, val, test

    def load(self):
        # loc = np.load('n_body_system/dataset/loc_' + self.suffix + '.npy')
        loc = np.load(self.data_root + "loc_" + self.phase + self.suffix + ".npy")
        # vel = np.load('n_body_system/dataset/vel_' + self.suffix + '.npy')
        vel = np.load(self.data_root + "vel_" + self.phase + self.suffix + ".npy")
        # edges = np.load('n_body_system/dataset/edges_' + self.suffix + '.npy')
        edges = np.load(self.data_root + "edges_" + self.phase + self.suffix + ".npy")

        # charges = np.load('n_body_system/dataset/charges_' + self.suffix + '.npy')
        charges = np.load(self.data_root + "charges_" + self.phase + self.suffix + ".npy")

        print("loc shape: ", loc.shape)
        print("vel shape: ", vel.shape)
        print("edges shape: ", edges.shape)
        print("charges shape: ", charges.shape)

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)

        return loc, vel, edge_attr, charges, edges

    def limit_samples(self, loc, vel, edges, charges):
        min_size = min(loc.size(0), self.max_samples)
        loc = loc[:min_size]
        vel = vel[:min_size]
        charges = charges[:min_size]
        edges = edges[:min_size]
        return loc, vel, edges, charges

    def preprocess(self, loc, vel, edges, charges):
        # Convert arrays to tensors and adjust dimension ordering
        loc = torch.tensor(loc).float().permute(0, 2, 3, 1)  # [batch, nodes, features, time_steps]
        vel = torch.tensor(vel).float().permute(0, 2, 3, 1)  # [batch, nodes, features, time_steps]
        charges = torch.tensor(charges).float()

        # Limit the number of samples if max_samples is set
        if self.max_samples is not None:
            loc, vel, edges, charges = self.limit_samples(loc, vel, edges, charges)

        # Make loc and vel translation invariant by subtracting the mean across nodes
        loc_mean = loc - loc.mean(dim=1, keepdim=True)
        vel_mean = vel - vel.mean(dim=1, keepdim=True)

        # Flatten from [batch, nodes, features, time_steps] to [batch * nodes * time_steps, features]
        loc_flat = loc_mean.reshape(-1, loc_mean.size(-1))
        vel_flat = vel_mean.reshape(-1, vel_mean.size(-1))

        # Handle edges
        edges, edge_attr = self.get_edges(edges)

        return loc_flat, vel_flat, edge_attr, charges, edges

    def get_edges(self,adjacency_matrices):
        batch_size, n_nodes, _ = adjacency_matrices.shape

        # Generate indices for all possible pairs of nodes
        rows, cols = np.meshgrid(np.arange(n_nodes), np.arange(n_nodes), indexing='ij')

        # Flatten the indices arrays and filter out self-loops
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]

        # Gather edge attributes
        edge_attr = adjacency_matrices[:, rows, cols]

        # Initialize the result tensor
        edges_tensor = np.stack((rows, cols))

        # Prepare the tensor with PyTorch
        edge_attr_tensor = torch.from_numpy(edge_attr).unsqueeze(2).float()
        edges_tensor = torch.tensor(edges_tensor, dtype=torch.int64)

        return edges_tensor, edge_attr_tensor

    def clifford_embedding(self):


class NBodyDataLoader():
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


n = NBodyDataset("", "train_charged5_initvel1small")
n.load()
