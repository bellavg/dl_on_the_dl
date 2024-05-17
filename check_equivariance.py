import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformer.transformer import NBodyTransformer
from algebra.cliffordalgebra import CliffordAlgebra
from dataset import NBody
from torch.utils.tensorboard import SummaryWriter




if __name__ == "__main__":
    metric = [1, 1, 1]
    clifford_algebra = CliffordAlgebra(metric)

    # Hyperparameters
    input_dim = 3  # feature_dim
    d_model = 16
    num_heads = 8
    num_layers = 6
    embed_in_features = 3
    embed_out_features = 3
    batch_size = 100
    num_samples = 1000

    # Create the model
    feature_embedding = NBodyTransformer(input_dim, d_model, num_heads, num_layers, clifford_algebra)

    random_nodes_pos = torch.randn((1, 5, 3))
    random_nodes_pos = random_nodes_pos - random_nodes_pos.mean(dim=1, keepdim=True)
    random_nodes_vel = torch.randn((1, 5, 3))
    random_nodes_vel = random_nodes_vel - random_nodes_vel.mean(dim=1, keepdim=True)
    random_nodes_charges = torch.randn((1, 5, 1))
    algebra = CliffordAlgebra((1,1,1))
    batch = [random_nodes_pos, random_nodes_vel, random_nodes_charges, random_nodes_pos]
    #loc, vel, edge_attr, charges, loc_end, edges
    res, _ = feature_embedding.forward(batch)#[..., 0, 1:4]

    rotor = algebra.versor(order=2)
    random_nodes_pos = random_nodes_pos.reshape(5, 3)
    random_nodes_vel = random_nodes_vel.reshape(5, 3)
    pos = algebra.embed_grade(random_nodes_pos.unsqueeze(1), 1)
    vel = algebra.embed_grade(random_nodes_vel.unsqueeze(1), 1)
    pos = algebra.rho(rotor, pos)[..., 1:4]
    vel = algebra.rho(rotor, vel)[..., 1:4]
    pos = pos.reshape(1, 5, 3)
    vel = vel.reshape(1, 5, 3)
    batch2 = [pos, vel, random_nodes_charges, pos]
    rot_res, _ = feature_embedding.forward(batch2)
    res = algebra.embed_grade(res.unsqueeze(dim=-2), 1)
    res_rot = algebra.rho(rotor, res)[..., 0, 1:4]
    print((res_rot - rot_res).sum())