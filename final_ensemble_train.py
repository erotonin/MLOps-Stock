import traceback
from ensemble_trainer import train_ensemble

def train_production_models():
    # Target symbols
    symbols = ["VNM", "VCB", "HPG", "FPT"]
    print(f"Starting Production Training for: {symbols}")
    
    failed = []
    
    for symbol in symbols:
        try:
            print(f"\nTraining Hybrid Ensemble for {symbol}...")
            train_ensemble(symbol=symbol, epochs=30)
            print(f"Completed {symbol}")
        except Exception as e:
            print(f"Error training {symbol}: {e}")
            traceback.print_exc()
            failed.append(symbol)

    if failed:
        print(f"\nTraining failed for: {failed}")
    else:
        print("\nAll models trained successfully! Artifacts saved to ./models/")

if __name__ == "__main__":
    train_production_models()
