import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from vn30_data import VN30Data
from vn30_model import FinCastMini

def calculate_winkler_score(y_true, q10, q90, alpha=0.2):
    """
    Winkler Score for 80% confidence interval (alpha=0.2).
    Lower is better.
    """
    scores = []
    for y, l, u in zip(y_true, q10, q90):
        width = u - l
        if y < l:
            score = width + (2/alpha) * (l - y)
        elif y > u:
            score = width + (2/alpha) * (y - u)
        else:
            score = width
        scores.append(score)
    return np.mean(scores)

def walk_forward_cv(symbol="FPT", n_folds=10, train_window=90, gap=5, test_window=10):
    print(f"\n--- Starting 10-fold Walk-forward CV for {symbol} ---")
    data_provider = VN30Data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = "fincast_vn30.pth"
    model = FinCastMini(input_size=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Fetch data (Total needed: train + gap + test * folds)
    total_days = train_window + gap + (test_window * n_folds)
    df = data_provider.get_historical_data(symbol, days=total_days + 30)
    if df is None: return None
    
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi']
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    results = []
    
    # Sliding window logic
    # Fold 0: [0:train_window] -> gap -> [train_window+gap : train_window+gap+test_window]
    for fold in range(n_folds):
        start_idx = fold * test_window
        train_end = start_idx + train_window
        test_start = train_end + gap
        test_end = test_start + test_window
        
        if test_end > len(scaled_data):
            break
            
        # Prepare test sequences
        X_fold = []
        y_true_fold_scaled = []
        
        for i in range(test_start, test_end):
            X_fold.append(scaled_data[i-60:i])
            y_true_fold_scaled.append(scaled_data[i, 3])
            
        X_fold = torch.FloatTensor(np.array(X_fold)).to(device)
        
        with torch.no_grad():
            quantiles = model(X_fold).cpu().numpy()
            
        # Inverse scale
        dummy = np.zeros((test_window, 8))
        
        # Q10
        dummy[:, 3] = quantiles[:, 0]
        q10 = scaler.inverse_transform(dummy)[:, 3]
        
        # Q50 (Median)
        dummy[:, 3] = quantiles[:, 1]
        q50 = scaler.inverse_transform(dummy)[:, 3]
        
        # Q10
        dummy[:, 3] = quantiles[:, 2]
        q90 = scaler.inverse_transform(dummy)[:, 3]
        
        # Actual
        dummy[:, 3] = y_true_fold_scaled
        y_true = scaler.inverse_transform(dummy)[:, 3]
        
        # Metrics for this fold
        mae = np.mean(np.abs(y_true - q50))
        rmse = np.sqrt(np.mean((y_true - q50)**2))
        coverage = np.mean((y_true >= q10) & (y_true <= q90))
        winkler = calculate_winkler_score(y_true, q10, q90)
        
        if fold == 0:
            print(f"DEBUG Fold 1 Sample: Actual={y_true[0]:.2f}, Q10={q10[0]:.2f}, Q50={q50[0]:.2f}, Q90={q90[0]:.2f}")
            print(f"DEBUG Spread: {np.mean(q90 - q10):.2f}")

        results.append({
            "Fold": fold + 1,
            "MAE": mae,
            "RMSE": rmse,
            "Coverage": coverage,
            "Winkler": winkler
        })
        print(f"Fold {fold+1}: MAE={mae:.2f}, Coverage={coverage*100:.1f}%")

    report_df = pd.DataFrame(results)
    print("\n" + "="*40)
    print(f"FINAL ACADEMIC REPORT FOR {symbol}")
    print("="*40)
    print(report_df.to_string(index=False))
    print("-" * 40)
    print(f"Mean MAE: {report_df['MAE'].mean():.2f}")
    print(f"Avg Coverage: {report_df['Coverage'].mean()*100:.1f}% (Goal: >= 78%)")
    print(f"Avg Winkler Score: {report_df['Winkler'].mean():.2f}")
    
    return report_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="FPT")
    args = parser.parse_args()
    walk_forward_cv(symbol=args.symbol)
