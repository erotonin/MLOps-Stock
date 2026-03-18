import torch
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from vn30_data import VN30Data
from vn30_model import FinCastMini
import matplotlib.pyplot as plt

def realtime_prediction_loop(symbol="FPT", window_size=60, update_interval=60):
    # 1. Setup
    data_provider = VN30Data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture
    model = FinCastMini(input_size=8).to(device)
    model_path = "fincast_vn30.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please run train.py first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"Starting real-time prediction for {symbol}...")
    print(f"Update interval: {update_interval} seconds")
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    actual_prices = []
    predicted_prices = []
    q10_list = []
    q90_list = []
    timestamps = []

    try:
        while True:
            # 2. Fetch Latest Data
            # Use 1m resolution for real-time tracking if market is open
            df = data_provider.get_historical_data(symbol, days=5, resolution='1m')
            
            # Fallback to daily if 1m is unavailable (e.g., market closed)
            if df is None or len(df) < window_size:
                df = data_provider.get_historical_data(symbol, days=100, resolution='D')
                
            if df is None or len(df) < window_size:
                print("Insufficient data, retrying...")
                time.sleep(10)
                continue
            
            # 3. Preprocess
            features = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi']
            data_values = df[features].values
            
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data_values)
            
            # Input sequence (last window_size rows)
            input_seq = scaled_data[-window_size:]
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            
            # 4. Inference
            with torch.no_grad():
                quantiles_scaled = model(input_tensor).cpu().numpy().flatten()
            
            # 5. Inverse Scale Prediction (Index 1 is median)
            dummy = np.zeros((3, len(features)))
            dummy[:, 3] = quantiles_scaled
            quantiles_actual = scaler.inverse_transform(dummy)[:, 3]
            prediction_actual = quantiles_actual[1]
            q10, q90 = quantiles_actual[0], quantiles_actual[2]
            
            # 6. Log and Update Plot
            current_close = df.iloc[-1]['close']
            current_time = datetime.now().strftime('%H:%M:%S')
            
            timestamps.append(current_time)
            actual_prices.append(current_close)
            predicted_prices.append(prediction_actual)
            
            # Store confidence bands
            q10_list.append(q10)
            q90_list.append(q90)
            
            # Keep only last 20 points
            if len(timestamps) > 20:
                timestamps.pop(0)
                actual_prices.pop(0)
                predicted_prices.pop(0)
                q10_list.pop(0)
                q90_list.pop(0)
            
            ax.clear()
            ax.plot(timestamps, actual_prices, label='Actual Price', marker='o', color='blue')
            ax.plot(timestamps, predicted_prices, label='Predicted (Median)', linestyle='--', marker='x', color='red')
            ax.fill_between(timestamps, q10_list, q90_list, color='red', alpha=0.2, label='80% Confidence Band')
            ax.set_title(f"Real-time Prediction for {symbol}")
            ax.set_ylabel("Price")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            
            print(f"[{current_time}] Actual: {current_close:.2f} | Predicted: {prediction_actual:.2f}")
            
            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("Real-time prediction stopped.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time VN30 Stock Prediction")
    parser.add_argument("--symbol", type=str, default="FPT", help="Stock symbol (e.g., FPT, TCB, HPG)")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")
    parser.add_argument("--window", type=int, default=60, help="Input window size")
    args = parser.parse_args()

    realtime_prediction_loop(symbol=args.symbol, update_interval=args.interval, window_size=args.window)
