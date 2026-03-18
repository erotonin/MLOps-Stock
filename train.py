import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from vn30_data import VN30Data
from vn30_model import FinCastMini

class StockDataset(Dataset):
    def __init__(self, data, window_size=60):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        return (
            self.data[idx : idx + self.window_size],
            self.data[idx + self.window_size, 3],  # Target: next Close price (index 3)
        )

class PQLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(PQLoss, self).__init__()
        self.quantiles = quantiles
        self.huber = nn.HuberLoss()

    def forward(self, pred_quantiles, target):
        """
        pred_quantiles: [batch, 3]
        target: [batch]
        """
        loss = 0
        for i, q in enumerate(self.quantiles):
            error = target - pred_quantiles[:, i]
            # Quantile loss
            q_loss = torch.max((q - 1) * error, q * error).mean()
            loss += q_loss
            
            # Point loss for the median (q=0.5)
            if q == 0.5:
                loss += self.huber(pred_quantiles[:, i], target)
        
        # Trend Consistency Loss
        # We assume the last 'Close' in the input sequence was the baseline for 'target'
        # But since we don't have it here easily, we use a simple penalty for 
        # large quantile overlaps or negative spreads (q90 must be > q10)
        spread_penalty = torch.relu(pred_quantiles[:, 0] - pred_quantiles[:, 2]).mean()
        loss += 0.1 * spread_penalty
        
        return loss

def train_fincast(symbols=None, epochs=30, window_size=60, lr=0.0005):
    data_provider = VN30Data()
    if symbols is None:
        symbols = data_provider.get_vn30_symbols()[:15] # Expand to 15 symbols for better generalization
    
    all_datasets = []
    scaler = MinMaxScaler()
    
    # 1. Fetch and Prepare Data for multiple symbols
    print(f"Fetching data for {len(symbols)} symbols...")
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi']
    
    for symbol in symbols:
        df = data_provider.get_historical_data(symbol, days=365)
        if df is not None and len(df) > window_size:
            data = df[features].values
            scaled_data = scaler.fit_transform(data) # In real app, use global or pre-calculated scaler
            all_datasets.append(StockDataset(scaled_data, window_size))
    
    if not all_datasets:
        print("No data available for training.")
        return

    full_dataset = ConcatDataset(all_datasets)
    dataloader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    
    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinCastMini(input_size=len(features)).to(device)
    criterion = PQLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Training Loop
    print(f"Starting FinCast training on {device} with {len(full_dataset)} samples...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x) # [batch, 3]
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.6f}")
            
    # 4. Save Model
    torch.save(model.state_dict(), "fincast_vn30.pth")
    print("Model saved to fincast_vn30.pth")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train FinCast-Mini for VN30")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="List of symbols to train on")
    args = parser.parse_args()
    
    train_fincast(symbols=args.symbols, epochs=args.epochs)
