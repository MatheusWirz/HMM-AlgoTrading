import pandas as pd
import numpy as np

# RSI (Relative Strength Index)

def compute_rsi(df: pd.DataFrame, window: int) -> pd.Series:

    # Close price
    close = df['close'].astype(float)

    # Take gain loss window
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Avarage gain and loss
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # If avg_loss==0, rsi=100
    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    rsi.name = f'RSI_{window}'

    return rsi