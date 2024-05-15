import torch
import torch.nn as nn
from algebra.cliffordalgebra import CliffordAlgebra
from original_models.modules.linear import MVLinear
from transformer.modules.clifford_embedding import NBodyGraphEmbedder
from transformer.modules.attention import GAST

# Define the metric for 3D space (Euclidean)
metric = [1, 1, 1]
d = len(metric)

# Initialize the Clifford Algebra for 3D
clifford_algebra = CliffordAlgebra(metric)


class NBodyTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, embed_in_features, embed_out_features,
                 clifford_algebra, channels):
        super(NBodyTransformer, self).__init__()
        self.embedding_layer = NBodyGraphEmbedder(clifford_algebra, in_features=input_dim, embed_dim=d_model)
        self.GAST = GAST(num_layers, d_model, num_heads, clifford_algebra, channels)
        self.embedding = MVLinear(clifford_algebra, embed_in_features, embed_out_features, subspaces=False)
        self.MV_input = MVLinear(clifford_algebra, input_dim, d_model, subspaces=True)
        self.MV_GP = MVLinear(clifford_algebra, d_model * 2, d_model, subspaces=True)

    def forward(self, batch):
        batch_size, n_nodes, _ = batch[0].size()

        # Generate node and edge embeddings along with the attention mask
        nodes, edges, loc_end_clifford, attention_mask = self.embedding_layer.embed_nbody_graphs(batch)

        # Combine nodes and edges after projection
        src = torch.cat((nodes, edges), dim=0)

        # Geometric Product
        src_GP = self.clifford_algebra.geometric_product(src, src)

        # Concatenate src and its geometric product
        src_cat = torch.cat((src, src_GP), dim=1)
        src = self.MVGP(src_cat)

        # Pass through GAST layers
        output = self.GAST(src, attention_mask)

        # Return only nodes and only the "pos" feature vector of the nodes
        return output[:(5 * batch_size), 1, :], loc_end_clifford
