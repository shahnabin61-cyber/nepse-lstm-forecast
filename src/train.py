import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import NepseDataset
from model import LSTMForecaster


def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(NepseDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NepseDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = LSTMForecaster().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        train_losses.append(total_train_loss / len(train_loader.dataset))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val_loss += loss.item() * xb.size(0)
        val_losses.append(total_val_loss / len(val_loader.dataset))

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.6f} - Val Loss: {val_losses[-1]:.6f}")

    return model, train_losses, val_losses