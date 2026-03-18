import requests
import time
import json
import os

BASE_URL = "http://localhost:8080"
USERNAME = "admin"
PASSWORD = "admin123"

def test_01_get_token():
    print("\n[TEST 01] Testing Token Generation (OAuth2/RS256)...")
    resp = requests.post(f"{BASE_URL}/token", data={"username": USERNAME, "password": PASSWORD})
    if resp.status_code == 200:
        token = resp.json().get("access_token")
        print(f"PASS: Token received (Type: {resp.json().get('token_type')})")
        return token
    else:
        print(f"FAIL: {resp.status_code} - {resp.text}")
        return None

def test_02_secure_prediction(token):
    print("\n[TEST 02] Testing Secure Prediction API (Layer 1 & 2 Anomalies)...")
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{BASE_URL}/predict/FPT", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        print(f"PASS: Received prediction for {data['symbol']}")
        print(f"      Current Price: {data['current_price']}")
        print(f"      Predicted Median: {data['predicted_median']}")
        print(f"      Anomalies Detected: {data['anomalies']['is_anomaly']}")
        if data['anomalies']['reasons']:
            print(f"      Reasons: {data['anomalies']['reasons']}")
        return True
    else:
        print(f"FAIL: {resp.status_code} - {resp.text}")
        return False

def test_03_rate_limiting(token):
    print("\n[TEST 03] Testing Rate Limiting (SlowAPI)...")
    headers = {"Authorization": f"Bearer {token}"}
    print("Calling API 10 times rapidly...")
    # Token endpoint has 5/min limit
    for i in range(7):
        resp = requests.post(f"{BASE_URL}/token", data={"username": USERNAME, "password": PASSWORD})
        if resp.status_code == 429:
            print(f"PASS: Rate limit triggered at call {i+1} (429 Too Many Requests)")
            return True
    print("WARNING: Rate limit not triggered. Check SlowAPI config.")
    return False

def test_04_audit_log_verification():
    print("\n[TEST 04] Verifying Persistent Audit Log...")
    if os.path.exists("audit_log.jsonl"):
        with open("audit_log.jsonl", "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                print(f"PASS: Audit log found with {len(lines)} entries.")
                last_entry = json.loads(lines[-1])
                print(f"      Last Entry User: {last_entry['user']} (Role: {last_entry['role']})")
                return True
    print("FAIL: Audit log file not found or empty.")
    return False

if __name__ == "__main__":
    print("=== FINCAST-MINI COMPREHENSIVE TEST SUITE ===")
    
    # 1. Auth
    token = test_01_get_token()
    
    if token:
        # 2. Prediction
        test_02_secure_prediction(token)
        
        # 3. Audit Log
        test_04_audit_log_verification()
        
        # 4. Rate Limit (Do last as it blocks)
        test_03_rate_limiting(token)
    else:
        print("CRITICAL: Skipping further tests due to Auth failure.")
