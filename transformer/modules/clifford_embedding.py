import torch
from original_models.modules.linear import MVLinear

from algebra.cliffordalgebra import CliffordAlgebra


class NBodyGraphEmbedder:
    def __init__(self, clifford_algebra, in_features, embed_dim):
        self.clifford_algebra = clifford_algebra
        self.node_projection = MVLinear(
            self.clifford_algebra, in_features, embed_dim, subspaces=False
        )

    def embed_nbody_graphs(self, batch):
        batch_size, n_nodes, _ = batch[0].size()
        full_embedding = self.get_embedding(batch)
        attention_mask = None
        return full_embedding, attention_mask

    def get_embedding(self, batch):
        loc_mean, vel, charges = self.preprocess(batch)

        # Embed data in Clifford space
        invariants = self.clifford_algebra.embed(charges, (0,))
        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.clifford_algebra.embed(xv, (1, 2, 3))

        nodes_stack = torch.cat([invariants[:, None], covariants], dim=1)
        full_node_embedding = self.node_projection(nodes_stack)
        return full_node_embedding

    def preprocess(self, batch):
        loc, vel, charges = batch
        loc_mean = self.compute_mean_centered(loc)

        loc_mean, vel, charges = self.flatten_tensors(loc_mean, vel, charges)
        return loc_mean, vel, charges

    def compute_mean_centered(self, tensor):
        return tensor - tensor.mean(dim=1, keepdim=True)

    def flatten_tensors(self, *tensors):
        return [tensor.float().view(-1, *tensor.shape[2:]) for tensor in tensors]
