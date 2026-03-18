import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from vn30_data import VN30Data
from vn30_model import FinCastMini

def get_metrics(symbol, window_size=60):
    data_provider = VN30Data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "fincast_vn30.pth"
    if not os.path.exists(model_path): return None
    
    model = FinCastMini(input_size=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    df = data_provider.get_historical_data(symbol, days=150)
    if df is None or len(df) < window_size + 20: return None
    
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi']
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    test_size = 30
    test_data = scaled_data[-(test_size + window_size):]
    X_test = []
    y_actual_scaled = []
    for i in range(test_size):
        X_test.append(test_data[i : i + window_size])
        y_actual_scaled.append(test_data[i + window_size, 3])
        
    X_test = torch.FloatTensor(np.array(X_test)).to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test)[:, 1].cpu().numpy().flatten()
        
    dummy_actual = np.zeros((test_size, len(features)))
    dummy_actual[:, 3] = y_actual_scaled
    y_actual = scaler.inverse_transform(dummy_actual)[:, 3]
    
    dummy_pred = np.zeros((test_size, len(features)))
    dummy_pred[:, 3] = y_pred_scaled
    y_pred = scaler.inverse_transform(dummy_pred)[:, 3]
    
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    return {"mae": mae, "rmse": rmse}

if __name__ == "__main__":
    symbols = ["FPT", "TCB", "VNM", "SSI", "HPG", "VIC", "VHM", "MSN", "MWG", "STB"]
    report = []
    for s in symbols:
        m = get_metrics(s)
        if m:
            report.append({"Symbol": s, "MAE": f"{m['mae']:.2f}", "RMSE": f"{m['rmse']:.2f}"})
    
    print("\nFINAL SYSTEM PERFORMANCE REPORT")
    print("="*30)
    df_report = pd.DataFrame(report)
    print(df_report.to_string(index=False))
