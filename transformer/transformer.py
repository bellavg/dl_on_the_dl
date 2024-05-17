import torch
import torch.nn as nn
from algebra.cliffordalgebra import CliffordAlgebra
from original_models.modules.linear import MVLinear
from transformer.modules.clifford_embedding import NBodyGraphEmbedder
from transformer.modules.attention import MainBody


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
        batch_size, n_nodes, _ = batch[0].size()


        # Generate node and edge embeddings along with the attention mask add back attention mask at smoe point please
        full_embeddings, loc_end_clifford, attention_mask = self.embedding_layer.embed_nbody_graphs(
            batch)

        # nodes -> [batch_size * n_nodes, d_model, 8]
        # edges -> [batch_size * n_edges, d_model, 8]

        #src = self.clifford_algebra.geometric_product(full_embeddings, full_embeddings)

        # src -> [batch_size * (n_nodes + n_edges), d_model, 8]

        # Apply MVLinear transformation to the combined embeddings
        src = self.combined_projection(full_embeddings)
        # src -> [batch_size * (n_nodes + n_edges), d_model*2, 8]

        # Pass through GAST layers
        output = self.GAST(src, attention_mask)

        return output[:(5 * batch_size), 1, :], loc_end_clifford
