import torch
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from sklearn.preprocessing import MinMaxScaler
from vn30_data import VN30Data
from vn30_model import FinCastMini
from security import verify_password, create_access_token, decode_access_token, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES

# --- 1. Pydantic Models (Strict Schema) ---
class Token(BaseModel):
    access_token: str
    token_type: str

class AnomalyInfo(BaseModel):
    is_anomaly: bool
    layer1_quantile: bool
    layer2_volatility: bool
    reasons: List[str]

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_median: float
    confidence_interval: Dict[str, float]
    anomalies: AnomalyInfo
    audit_status: str
    timestamp: str

# --- 2. Setup App, Security & Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="VN30 FinCast-Mini Secure API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("admin123"),
        "role": "admin"
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": get_password_hash("analyst123"),
        "role": "analyst"
    }
}

# --- 3. Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FinCastMini(input_size=8).to(device)
model_path = "fincast_vn30.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

data_provider = VN30Data()

# --- 4. Dependencies ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token (RS256 verification failed)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username: str = payload.get("sub")
    user = FAKE_USERS_DB.get(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def verify_data_integrity(df: pd.DataFrame):
    # Academic mock: In production, this would check checksums from vnstock
    if df is None or df.empty:
        return False
    return True

# --- 5. Endpoints ---
@app.post("/token", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/predict/{symbol}", response_model=PredictionResponse)
@limiter.limit("60/minute")
async def predict(request: Request, symbol: str, current_user: dict = Depends(get_current_user)):
    # 1. Fetch & Verify Integrity
    df = data_provider.get_historical_data(symbol, days=5, resolution='1m')
    if df is None or len(df) < 60:
        df = data_provider.get_historical_data(symbol, days=100, resolution='D')
    
    if not verify_data_integrity(df):
        raise HTTPException(status_code=404, detail="Data integrity check failed or insufficient data")
    
    # 2. Preprocess
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi']
    data_values = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_values)
    
    input_seq = scaled_data[-60:]
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
    
    # 3. Inference
    with torch.no_grad():
        quantiles_scaled = model(input_tensor).cpu().numpy().flatten()
    
    # 4. Inverse Scale
    dummy = np.zeros((3, len(features)))
    dummy[:, 3] = quantiles_scaled
    quantiles_actual = scaler.inverse_transform(dummy)[:, 3]
    
    prediction = float(quantiles_actual[1])
    q10, q90 = float(quantiles_actual[0]), float(quantiles_actual[2])
    current_price = float(df.iloc[-1]['close'])
    
    # 5. Anomaly Detection (Two-Layer)
    is_anomaly_l1 = current_price < q10 or current_price > q90
    
    last_10_prices = df['close'].tail(10).values
    returns = np.diff(last_10_prices)
    z_score = abs((current_price - df.iloc[-2]['close'] - np.mean(returns)) / (np.std(returns) + 1e-6)) if len(returns) > 0 else 0
    is_anomaly_l2 = z_score > 3.0
    
    final_anomaly = is_anomaly_l1 or is_anomaly_l2
    reasons = []
    if is_anomaly_l1: reasons.append("Layer 1: Q-Band Breach")
    if is_anomaly_l2: reasons.append(f"Layer 2: High Volatility (Z={z_score:.2f})")
    
    # 6. Persistent Audit Log
    log_entry = {
        "at": datetime.now().isoformat(),
        "user": current_user['username'],
        "role": current_user['role'],
        "symbol": symbol,
        "anomaly": final_anomaly,
        "reasons": reasons,
        "ip": request.client.host
    }
    with open("audit_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "predicted_median": prediction,
        "confidence_interval": {"q10": q10, "q90": q90},
        "anomalies": {
            "is_anomaly": final_anomaly,
            "layer1_quantile": is_anomaly_l1,
            "layer2_volatility": is_anomaly_l2,
            "reasons": reasons
        },
        "audit_status": "logged_to_append_only_file",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
