import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformer.transformer import NBodyTransformer
from algebra.cliffordalgebra import CliffordAlgebra
from dataset import NBody
import optuna
import joblib


def objective(trial):
    # Define search space for hyperparameters
    input_dim = 3
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 1, 8)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    wd = trial.suggest_loguniform('wd', 1e-5, 1e-3)
    dp = trial.suggest_uniform('dp', 0.1, 0.5)


    clifford_algebra = CliffordAlgebra([1, 1, 1])

    num_samples = 1000

    # Create the model
    model = NBodyTransformer(input_dim, d_model, num_heads, num_layers, clifford_algebra, dropout=dp)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    nbody_data = NBody(num_samples=num_samples, batch_size=batch_size)
    train_loader = nbody_data.train_loader()
    val_loader = nbody_data.val_loader()

    def train_epoch(model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            output, tgt = model(batch)
            loss = criterion(output, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    epochs = 50
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)

    # save study to file
    study_name = "nbody_study"
    joblib.dump(study, "study.pkl")

