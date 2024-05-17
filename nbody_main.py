import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformer.transformer import NBodyTransformer
from algebra.cliffordalgebra import CliffordAlgebra
from dataset import NBody
from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        output, tgt = model(batch)
        loss = criterion(output, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            output, tgt = model(batch)
            loss = criterion(output, tgt)
            running_loss += loss.item()
    return running_loss / len(val_loader)


metric = [1, 1, 1]
clifford_algebra = CliffordAlgebra(metric)

# Hyperparameters
input_dim = 3  # feature_dim
d_model = 16
num_heads = 8
num_layers = 6
embed_in_features = 3
embed_out_features = 3
batch_size = 1
num_samples = 1

# Create the model
model = NBodyTransformer(input_dim, d_model, num_heads, num_layers, clifford_algebra)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#optimizer = optim.Adam(model.parameters(), lr=0.001)  # Check is weight decay equivariant i feel like no...
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
T_0 = 10  # Number of iterations for the first restart
T_mult = 1  # A factor to increase T_i after a restart
eta_min = 0.00001  # Minimum learning rate

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min)

nbody_data = NBody(num_samples=num_samples, batch_size=batch_size)

train_loader = nbody_data.train_loader()
val_loader = nbody_data.val_loader()  # Assuming you have a validation data loader

best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_limit = 10

for epoch in tqdm(range(100)):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate_epoch(model, val_loader, criterion)
    scheduler.step(epoch)  # Update learning rate based on validation loss

    # Save model if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Stop training if validation loss has not improved for early_stopping_limit epochs
    if early_stopping_counter >= early_stopping_limit:
        print('Early stopping...')
        break

    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

# Ideas to try: add geometric product in self attention, add specific positional encoding
# Add dropout
