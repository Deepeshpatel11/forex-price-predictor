import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OANDA_API_KEY = os.getenv("OANDA_API_KEY")

# Initialize OANDA client (Practice Account)
client = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")

# Map timeframe to OANDA granularities
TIMEFRAME_MAP = {
    "M15": "M15",  # 15-minute
    "H1": "H1",    # 1-hour
    "H4": "H4",    # 4-hour
    "D": "D"       # 1-day
}

def fetch_live_data(pair: str, candles: int = 200, timeframe: str = "H1") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for the given forex pair from OANDA API.
    Supports fetching more than 5000 candles by batching requests.
    
    Args:
        pair (str): Forex pair in OANDA format, e.g., 'EUR_USD'
        candles (int): Total number of candles to fetch
        timeframe (str): One of 'M15', 'H1', 'H4', 'D'

    Returns:
        pd.DataFrame: timestamp, open, high, low, close, volume
    """
    granularity = TIMEFRAME_MAP.get(timeframe, "H1")
    all_data = []
    remaining = candles
    to_time = None  # For paging backwards in time

    try:
        while remaining > 0:
            batch_size = min(remaining, 5000)  # OANDA limit
            params = {
                "count": batch_size,
                "granularity": granularity,
                "price": "M"  # Mid prices
            }
            if to_time:
                params["to"] = to_time  # Fetch older data before this timestamp

            # Make API request
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            client.request(r)
            data = r.response.get("candles", [])

            if not data:
                break

            # Convert response to DataFrame
            df = pd.DataFrame([
                {
                    "timestamp": c["time"],
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c["volume"])
                }
                for c in data if c["complete"]
            ])

            all_data.append(df)

            # Prepare for next batch (oldest timestamp of this batch)
            to_time = data[0]["time"]
            remaining -= batch_size

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
            df_all.sort_values("timestamp", inplace=True)
            df_all.reset_index(drop=True, inplace=True)
            return df_all

        return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error fetching data for {pair}: {e}")
        return pd.DataFrame()  # Return empty DataFrame if failed
