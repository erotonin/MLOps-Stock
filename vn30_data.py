from vnstock import Vnstock
import pandas as pd
from datetime import datetime, timedelta
from indicators import add_technical_indicators

class VN30Data:
    def __init__(self):
        self.stock_client = Vnstock()

    def get_vn30_symbols(self):
        """
        Retrieves the list of VN30 constituent symbols.
        """
        try:
            # Correct call for vnstock v3.x
            df = self.stock_client.stock().listing.symbols_by_group('VN30')
            return df['ticker'].tolist()
        except Exception as e:
            print(f"Error fetching VN30 list: {e}")
            # Fallback
            return [
                "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
                "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
                "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
            ]

    def get_historical_data(self, symbol, days=365, resolution='D'):
        """
        Fetches historical OHLCV data for a specific symbol.
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        try:
            # Correct call for vnstock v3.x
            # Quote.history(start, end, interval)
            quote = self.stock_client.stock(symbol).quote
            df = quote.history(start=start_date, end=end_date, interval=resolution)
            
            if df is not None and not df.empty:
                df = add_technical_indicators(df)
                return df
            return None
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None

    def get_intraday_data(self, symbol, resolution='1m'):
        """
        Fetches intraday data (e.g., 1-minute resolution) for real-time tracking.
        """
        # For intraday, we use the same history method but with smaller interval
        end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start_date = (datetime.now() - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            quote = self.stock_client.stock(symbol).quote
            df = quote.history(start=start_date, end=end_date, interval=resolution)
            
            if df is not None and not df.empty:
                df = add_technical_indicators(df)
                return df
            return None
        except Exception as e:
            print(f"Error fetching intraday data for {symbol}: {e}")
            return None

if __name__ == "__main__":
    # Quick test
    data_provider = VN30Data()
    vn30_list = data_provider.get_vn30_symbols()
    print(f"VN30 Symbols: {vn30_list[:5]}... (Total: {len(vn30_list)})")
    
    # Test fetch for one symbol
    if vn30_list:
        test_symbol = vn30_list[0]
        print(f"Fetching data for {test_symbol}...")
        df = data_provider.get_historical_data(test_symbol, days=5)
        if df is not None:
            print(df.head())
        else:
            print("Failed to fetch data.")
