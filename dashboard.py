from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import torch
import numpy as np
import joblib
import os
from yahoo_data import YahooData
from tft_model import TFTSkeleton
from decision_policy import DecisionContext, build_decision

app = FastAPI()
templates = Jinja2Templates(directory="templates")
data_provider = YahooData()

NUM_FEATURES = 13
FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 
    'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal', 
    'bb_upper', 'bb_lower', 'log_return'
]

MODELS_DIR = "./models"

def get_prediction(symbol):
    sym = symbol.upper()
    # Check if model files exist
    if not os.path.exists(os.path.join(MODELS_DIR, f"{sym}_tft_model.pt")):
        return {
            "current_price": 0,
            "predicted_price": 0,
            "trend": "N/A",
            "action": "HOLD",
            "confidence": 0.0,
            "expected_return_pct": 0.0,
            "reason": f"No local model found for {sym}",
        }
    
    df = data_provider.get_historical_data(symbol, days=200)
    if df is None or len(df) < 60:
        return {
            "current_price": 0,
            "predicted_price": 0,
            "trend": "N/A",
            "action": "HOLD",
            "confidence": 0.0,
            "expected_return_pct": 0.0,
            "reason": "Insufficient input data",
        }
    
    X = df[FEATURES].values
    last_price = df['close'].iloc[-1]
    realized_vol_pct = float(df['log_return'].tail(20).std() * 100.0) if 'log_return' in df.columns else 0.0
    
    # Load ALL artifacts from local MODELS_DIR
    scaler_x = joblib.load(os.path.join(MODELS_DIR, f"{sym}_scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, f"{sym}_scaler_y.pkl"))
    lgbm_model = joblib.load(os.path.join(MODELS_DIR, f"{sym}_lgbm_model.pkl"))
    meta_learner = joblib.load(os.path.join(MODELS_DIR, f"{sym}_meta_learner.pkl"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tft = TFTSkeleton(num_features=NUM_FEATURES)
    tft.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{sym}_tft_model.pt"), map_location=device, weights_only=True))
    tft.to(device).eval()
    
    X_scaled = scaler_x.transform(X)
    
    # Predict
    input_seq = torch.FloatTensor(X_scaled[-60:]).unsqueeze(0).to(device)
    with torch.no_grad():
        tft_p = tft(input_seq).item()
    lgbm_p = lgbm_model.predict(X_scaled[-1].reshape(1, -1))[0]
    
    # Stacking
    meta_X = np.column_stack([[tft_p], [lgbm_p]])
    ensemble_p_scaled = meta_learner.predict(meta_X)[0]
    ensemble_p = scaler_y.inverse_transform([[ensemble_p_scaled]])[0][0]
    tft_price = scaler_y.inverse_transform([[tft_p]])[0][0]
    lgbm_price = scaler_y.inverse_transform([[lgbm_p]])[0][0]
    uncertainty_pct = abs(float(tft_price) - float(lgbm_price)) / max(float(last_price), 1e-8) * 100.0
    
    trend = "UP" if ensemble_p > last_price else "DOWN"
    decision = build_decision(
        DecisionContext(
            current_price=float(last_price),
            predicted_price=float(ensemble_p),
            uncertainty_pct=float(uncertainty_pct),
        ),
        realized_volatility_pct=realized_vol_pct,
    )
    
    return {
        "current_price": round(float(last_price), 2),
        "predicted_price": round(float(ensemble_p), 2),
        "trend": trend,
        "action": decision.action,
        "confidence": round(float(decision.confidence), 4),
        "expected_return_pct": round(float(decision.expected_return_pct), 4),
        "reason": decision.reason,
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    symbols = ["VNM", "VCB", "HPG", "FPT"]
    predictions = {}
    for s in symbols:
        try:
            predictions[s] = get_prediction(s)
            predictions[s]["status"] = "ok"
        except Exception as e:
            print(f"Error predicting {s}: {e}")
            predictions[s] = {
                "status": "error",
                "current_price": "ERR",
                "predicted_price": "ERR",
                "trend": "N/A",
                "action": "HOLD",
                "confidence": 0,
                "expected_return_pct": 0,
                "reason": str(e),
            }
            
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "predictions": predictions
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
