import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Adds RSI and SMA indicators to the dataframe.
    Expects columns: 'close'
    """
    if df is None or len(df) < 14:
        return df
    
    # 1. Simple Moving Averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Fill NaNs from rolling windows
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df
