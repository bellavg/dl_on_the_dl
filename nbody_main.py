import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformer.transformer import NBodyTransformer
from algebra.cliffordalgebra import CliffordAlgebra
from dataset import NBody

# Define the metric for 3D space (Euclidean)
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
channels = 7
num_samples = 1000

# Create the model
model = NBodyTransformer(input_dim, d_model, num_heads, num_layers, clifford_algebra, channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

nbody_data = NBody(num_samples=num_samples, batch_size=batch_size)

train_loader = nbody_data.train_loader()

model.train()
for epoch in tqdm(range(10)):
    for i, batch in enumerate(train_loader):
        #print(f"Batch {i + 1} size: {batch[0].size()}")  # Assuming batch[0] is your data

        optimizer.zero_grad()
        output, tgt = model(batch)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')