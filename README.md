# MLOps Stock Hybrid Ensemble (Hiếu-Hoàng Thesis)

Building a Stock Price Prediction System Based on an MLOps Architecture with Ensemble Models on a Hybrid Cloud Platform.

## 🚀 Key Features
- **Hybrid Ensemble Model (Phase 5)**: Combining **Temporal Fusion Transformer (TFT)** and **LightGBM** via a **Stacking Meta-Learner**. 
- **High Performance**: Achieving **82.8% average directional accuracy** for T+1 forecasting on VN30 stocks (VNM, VCB, HPG, FPT).
- **MLOps Lifecycle**: Experiment tracking and artifact management via **MLflow**.
- **Serving Governance**: Champion pinning + artifact manifest integrity validation before serving.
- **DevSecOps**: Secure API with **RS256 Asymmetric JWT** and persistent **Audit Logging**.
- **Data Pipeline**: Automated daily fetching from **Yahoo Finance** with 13 technical indicators.
- **Release Gate Hardening**: Go/No-Go combines offline metrics, multi-window walk-forward stability, and decision backtest risk checks.
- **Web Dashboard**: Modern Glassmorphism UI for real-time trend monitoring.

## 📦 Project Structure
- `app.py`: Secure FastAPI for prediction inference.
- `dashboard.py`: FastAPI server for the Web Dashboard.
- `ensemble_trainer.py`: Core training module with Stacking logic.
- `tft_model.py` / `lgbm_model.py`: Model architecture definitions.
- `yahoo_data.py` / `indicators.py`: Data fetching and feature engineering.
- `monitor_drift.py`: K-S test for data drift detection.
- `security.py`: JWT and RS256 security utilities.

## 🛠 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables (Security + MLflow)**:
   ```bash
   # Required in production
   export APP_ADMIN_USERNAME=admin
   export APP_ADMIN_PASSWORD=<strong_password>
   export APP_ENV=production
   export APP_ALLOW_INSECURE_DEFAULTS=false
   export PRIVATE_KEY_PEM='<your_private_key_pem_content>'
   export PUBLIC_KEY_PEM='<your_public_key_pem_content>'
   export JWT_ISSUER=nt214-mlops
   export JWT_AUDIENCE=nt214-api

   # Optional (default uses sqlite:///mlflow.db)
   export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

   # Optional serving gates
   export MIN_SERVE_DIRECTIONAL_ACC=0
   export MAX_SERVE_VAL_RMSE=1000000000
   ```

3. **Run Web Dashboard**:
   ```bash
   python dashboard.py
   ```
   Access at `http://localhost:8081`

4. **Run Secure API**:
   ```bash
   python app.py
   ```
   Endpoint: `GET /predict/{symbol}` (Requires JWT Bearer Token)

5. **Backtest HOLD/BUY/SELL Decisions**:
   ```bash
   python backtest_decision.py
   ```

6. **Run Artifact Contract Check**:
   ```bash
   python artifact_contract_check.py
   ```

7. **Run Go/No-Go Release Gate**:
   ```bash
   python go_no_go_check.py
   ```

8. **Promote Champion Run (optional but recommended)**:
   ```bash
   python promote_champion.py --symbol FPT --run-id <mlflow_run_id>
   ```

9. **Full System Audit**:
   ```bash
   python run_full_audit.py
   ```

10. **Experiment Tracking**:
   View MLflow UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

## ✅ Go/No-Go Criteria (Current)
- Artifact contract completeness + finished run.
- Directional accuracy / action precision / action coverage range / action macro-F1.
- Walk-forward stability (mean accuracy, accuracy std, mean macro-F1).
- Decision backtest risk metrics (portfolio return, max drawdown, profit factor, Sharpe).

## 🔒 Security Note
- Private keys (`private_key.pem`) are stored locally. Do NOT commit to Git.
- Audit logs are recorded in `audit_log.jsonl`.

---
**Authors**: Lê Đình Hiếu, Trần Việt Hoàng (ATTT2023.1)  
**Advisor**: ThS. Lê Anh Tuấn
