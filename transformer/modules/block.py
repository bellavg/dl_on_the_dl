import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from original_models.modules.linear import MVLinear
from original_models.modules.mvlayernorm import MVLayerNorm
from original_models.modules.mvsilu import MVSiLU
from einops import rearrange
from transformer.modules.attention import SelfAttentionClifford


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, clifford_algebra):
        super(TransformerBlock, self).__init__()
        self.algebra = clifford_algebra
        self.mvlayernorm1 = MVLayerNorm(clifford_algebra, d_model)
        self.self_attn = SelfAttentionClifford(d_model, 5, clifford_algebra, num_heads)
        self.mvlayernorm2 = MVLayerNorm(clifford_algebra, d_model)
        self.mvlayernorm3 = MVLayerNorm(clifford_algebra, d_model)
        self.mvlayernorm4 = MVLayerNorm(clifford_algebra, d_model)
        self.mlp = nn.Sequential(
            MVLinear(clifford_algebra, d_model, d_model * 2),
            MVSiLU(clifford_algebra, d_model * 2),
            MVLinear(clifford_algebra, d_model * 2, d_model)
        )
        # self.dropout = TBD

    def forward(self, src, src_mask=None):
        # src -> [batch_size * (n_nodes + n_edges), d_model*2, 8]
        # Norm
        src_norm1 = self.mvlayernorm1(src)
        # Self-attention
        attended_src = self.self_attn(src_norm1, src_mask)

        # Add and norm
        src = src + attended_src
        src = self.mvlayernorm2(src)

        # # geo prod - possibly to take out - compare
        src_gp = self.algebra.geometric_product(src, src)
        src = src + src_gp
        src = self.mvlayernorm4(src)

        # MLP
        ff_src = self.mlp(src)
        # ff_src = self.dropout(ff_src)

        # Add and norm
        src = src + ff_src
        src = self.mvlayernorm3(src)

        return src


class MainBody(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, clifford_algebra):
        super(MainBody, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, clifford_algebra) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
