import pandas as pd
import numpy as np

# --------------------------
# CLEANING FUNCTION
# --------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data:
    - Remove duplicates
    - Sort by timestamp
    - Forward-fill missing values if any
    """
    df = df.copy()
    
    # Sort by timestamp and drop duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    
    # Handle missing values
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    return df

# --------------------------
# FEATURE ENGINEERING
# --------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for forex OHLC data.
    
    Returns:
        DataFrame with new columns: SMA50, SMA200, RSI, MACD, ATR
    """
    df = df.copy()
    
    # SMA50 and SMA200
    df['SMA50'] = df['close'].rolling(50).mean()
    df['SMA200'] = df['close'].rolling(200).mean()
    
    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12 EMA - 26 EMA)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # ATR (Average True Range, 14)
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = (df['high'] - df['close'].shift()).abs()
    df['L-C'] = (df['low'] - df['close'].shift()).abs()
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # Drop helper columns
    df.drop(['H-L','H-C','L-C','TR'], axis=1, inplace=True)
    
    return df

# --------------------------
# ADD TARGET COLUMN
# --------------------------
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary target column:
    1 if next close > current close, else 0
    """
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df
