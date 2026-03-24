import uvicorn
import torch
import os
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI
from yahoo_data import YahooData
from tft_model import TFTSkeleton
from decision_policy import DecisionContext, build_decision

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_t3: float
    expected_return_pct: float
    confidence: float
    action: str
    decision_reason: str
    trend: str
    timestamp: str

app = FastAPI(title="Simplified Stock Prediction API")
data_provider = YahooData()

NUM_FEATURES = 13
FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 
    'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal', 
    'bb_upper', 'bb_lower', 'log_return'
]
MODELS_DIR = "./models"
MODEL_CACHE: Dict[str, Dict[str, Any]] = {}

def _load_models_for_symbol(symbol: str) -> Dict[str, Any]:
    sym = symbol.upper()
    if sym in MODEL_CACHE:
        return MODEL_CACHE[sym]

    scaler_x = joblib.load(os.path.join(MODELS_DIR, f"{sym}_scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, f"{sym}_scaler_y.pkl"))
    lgbm_model = joblib.load(os.path.join(MODELS_DIR, f"{sym}_lgbm_model.pkl"))
    meta_learner = joblib.load(os.path.join(MODELS_DIR, f"{sym}_meta_learner.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tft = TFTSkeleton(num_features=NUM_FEATURES)
    tft.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, f"{sym}_tft_model.pt"), map_location=device, weights_only=True)
    )
    tft.to(device).eval()

    loaded = {
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "lgbm_model": lgbm_model,
        "meta_learner": meta_learner,
        "tft": tft,
        "device": device
    }
    MODEL_CACHE[sym] = loaded
    return loaded

def perform_ensemble_inference(symbol):
    sym = symbol.upper()
    df = data_provider.get_historical_data(symbol, days=200)
    if df is None or len(df) < 60:
        return None
    
    X = df[FEATURES].values
    last_price = df['close'].iloc[-1]
    realized_vol_pct = float(df['log_return'].tail(20).std() * 100.0) if 'log_return' in df.columns else 0.0
    
    loaded = _load_models_for_symbol(sym)
    scaler_x = loaded["scaler_x"]
    scaler_y = loaded["scaler_y"]
    lgbm_model = loaded["lgbm_model"]
    meta_learner = loaded["meta_learner"]
    tft = loaded["tft"]
    device = loaded["device"]
    
    X_scaled = scaler_x.transform(X)
    
    # Predict
    input_seq = torch.FloatTensor(X_scaled[-60:]).unsqueeze(0).to(device)
    with torch.no_grad():
        tft_p = tft(input_seq).item()
    
    lgbm_p = lgbm_model.predict(X_scaled[-1].reshape(1, -1))[0]
    
    # Stacking
    meta_X = np.column_stack([[tft_p], [lgbm_p]])
    ensemble_p_scaled = meta_learner.predict(meta_X)[0]
    
    # Inverse scale
    ensemble_p = scaler_y.inverse_transform([[ensemble_p_scaled]])[0][0]
    tft_price = scaler_y.inverse_transform([[tft_p]])[0][0]
    lgbm_price = scaler_y.inverse_transform([[lgbm_p]])[0][0]
    uncertainty_pct = abs(tft_price - lgbm_price) / max(float(last_price), 1e-8) * 100.0

    return {
        "symbol": sym,
        "current_price": float(last_price),
        "predicted_t3": float(ensemble_p),
        "uncertainty_pct": float(uncertainty_pct),
        "realized_volatility_pct": realized_vol_pct
    }

@app.get("/predict/{symbol}", response_model=PredictionResponse)
async def predict(symbol: str):
    inference = perform_ensemble_inference(symbol.upper())
    if inference is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Model or data not found for symbol")

    current_price = inference["current_price"]
    predicted_t3 = inference["predicted_t3"]
    trend = "UP" if predicted_t3 > current_price else "DOWN"

    decision = build_decision(
        DecisionContext(
            current_price=current_price,
            predicted_price=predicted_t3,
            uncertainty_pct=inference["uncertainty_pct"],
        ),
        realized_volatility_pct=float(inference.get("realized_volatility_pct", 0.0)),
    )
    
    return {
        "symbol": symbol.upper(),
        "current_price": current_price,
        "predicted_t3": predicted_t3,
        "expected_return_pct": round(decision.expected_return_pct, 4),
        "confidence": round(decision.confidence, 4),
        "action": decision.action,
        "decision_reason": decision.reason,
        "trend": trend,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
