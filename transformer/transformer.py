import torch
import torch.nn as nn
from algebra.cliffordalgebra import CliffordAlgebra
from original_models.modules.linear import MVLinear
from transformer.modules.clifford_embedding import NBodyGraphEmbedder
from transformer.modules.block import MainBody


class NBodyTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers,
                 clifford_algebra):
        super(NBodyTransformer, self).__init__()

        # Initialize the transformer with the given parameters
        # and the Clifford algebra
        self.clifford_algebra = clifford_algebra
        # Initialize the embedding layer
        self.embedding_layer = NBodyGraphEmbedder(self.clifford_algebra, in_features=input_dim, embed_dim=d_model)
        self.GAST = MainBody(num_layers, d_model, num_heads, self.clifford_algebra)
        self.combined_projection = MVLinear(self.clifford_algebra, d_model, d_model, subspaces=True)
        self.MV_input = MVLinear(self.clifford_algebra, input_dim, d_model, subspaces=True)
        self.MV_GP = MVLinear(self.clifford_algebra, d_model * 2, d_model, subspaces=True)

    def forward(self, batch):
        loc_start = batch[0]
        batch_size, n_nodes, _ = loc_start.size()

        loc_end = batch[-1]
        new_batch = batch[:-1]

        # Generate node and edge embeddings along with the attention mask add back attention mask at smoe point please
        full_embeddings, attention_mask = self.embedding_layer.embed_nbody_graphs(new_batch)

        src = self.clifford_algebra.geometric_product(full_embeddings, full_embeddings)


        # Pass through GAST layers
        output = self.GAST(src, attention_mask)
        output_locations = output[:(5 * batch_size), 1, 1:4]
        new_pos = loc_start + output_locations.view(batch_size, 5, 3)

        return new_pos, loc_end
