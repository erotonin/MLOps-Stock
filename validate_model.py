import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from vn30_data import VN30Data
from vn30_model import FinCastMini
import matplotlib.pyplot as plt

def validate_model(symbol="FPT", window_size=60):
    # 1. Setup
    data_provider = VN30Data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = "fincast_vn30.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please run train.py first.")
        return
        
    model = FinCastMini(input_size=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Fetch Data (Fetch more to have a test set)
    print(f"Fetching data for validation: {symbol}")
    df = data_provider.get_historical_data(symbol, days=200)
    if df is None or len(df) < window_size + 20:
        print("Insufficient data for validation.")
        return
        
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi']
    data = df[features].values
    
    # We must use the same scaling logic as training
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 3. Prepare Test Sequences
    # We take the last 50 samples for testing
    test_size = 50
    test_data = scaled_data[-(test_size + window_size):]
    
    X_test = []
    y_test_actual_scaled = []
    
    for i in range(test_size):
        X_test.append(test_data[i : i + window_size])
        y_test_actual_scaled.append(test_data[i + window_size, 3]) # Close
        
    X_test = torch.FloatTensor(np.array(X_test)).to(device)
    
    # 4. Run Inference
    print("Running validation inference...")
    with torch.no_grad():
        quantiles_scaled = model(X_test).cpu().numpy() # [test_size, 3]
        y_pred_scaled = quantiles_scaled[:, 1] # Use median for metrics
        q10_scaled = quantiles_scaled[:, 0]
        q90_scaled = quantiles_scaled[:, 2]
    
    # 5. Inverse Scaling
    # Actual
    dummy_actual = np.zeros((test_size, len(features)))
    dummy_actual[:, 3] = y_test_actual_scaled
    y_actual = scaler.inverse_transform(dummy_actual)[:, 3]
    
    # Predicted
    dummy_pred = np.zeros((test_size, len(features)))
    dummy_pred[:, 3] = y_pred_scaled
    y_pred = scaler.inverse_transform(dummy_pred)[:, 3]
    
    # 6. Calculate Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n--- Validation Results ---")
    print(f"Symbol: {symbol}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # 7. Plot Results
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual Price', color='blue', marker='o')
    plt.plot(y_pred, label='Predicted Price', color='red', linestyle='--', marker='x')
    plt.title(f"Validation: Actual vs Predicted Prices for {symbol}")
    plt.xlabel("Days (Recent)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate FinCast-Mini on a symbol")
    parser.add_argument("--symbol", type=str, default="FPT", help="Stock symbol to validate")
    args = parser.parse_args()
    
    validate_model(symbol=args.symbol)
