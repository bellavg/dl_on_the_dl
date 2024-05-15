import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from algebra.cliffordalgebra import CliffordAlgebra
from models.modules.linear import MVLinear
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.mvlayernorm import MVLayerNorm
from models.modules.mvsilu import MVSiLU
from models.nbody_cggnn import CEMLP
from data import nbody
import math
from clifford_embedding import NBodyGraphEmbedder

# Define the metric for 3D space (Euclidean)
metric = [1, 1, 1]
d = len(metric)

# Initialize the Clifford Algebra for 3D
clifford_algebra = CliffordAlgebra(metric)


class NBodyTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, batch_size, embed_in_features, embed_out_features,
                 clifford_algebra, channels):
        super(NBodyTransformer, self).__init__()
        self.embedding = NBodyGraphEmbedder(clifford_algebra, make_edge_attr=True)
        self.GAST = GAST(num_layers, d_model, num_heads, clifford_algebra, channels)
        self.embedding = MVLinear(clifford_algebra, embed_in_features, embed_out_features, subspaces=False)
        self.MVinput = MVLinear(clifford_algebra, input_dim, d_model, subspaces=True)
        self.MVGP = MVLinear(clifford_algebra, d_model * 2, d_model, subspaces=True)

    def forward(self, batch, batch_size):
        # Generate node and edge embeddings along with the attention mask
        nodes, edges, loc_end_clifford, attention_mask = self.embedding.embed_nbody_graphs(batch)

        # Position Encoding (if needed, uncomment and use)
        # edges_in_clifford = self.positional_encoding(edges_in_clifford)

        # Combine nodes and edges
        src = torch.cat((nodes, edges), dim=0)

        # Initial MV Linear Transformation
        src_MV = self.MVinput(src)
        src_GP = clifford_algebra.geometric_product(src_MV, src_MV)
        src_cat = torch.cat((src_MV, src_GP), dim=1)
        src = self.MVGP(src_cat)

        # Pass through GAST layers
        enc_output = self.GAST(src, attention_mask)
        output = enc_output

        # Return only nodes and only the "pos" feature vector of the nodes
        return output[:(5 * batch_size), 1, :]
