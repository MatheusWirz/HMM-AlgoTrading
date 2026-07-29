import numpy as np
import pandas as pd

# CMF (Chaikin Money Flow)

def compute_cmf(df: pd.DataFrame, window: int) -> pd.Series:

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    denom = (high - low).replace(0, np.nan)
    mfm = (2 * close - high - low) / denom
    mfm = mfm.fillna(0.0)

    mfv = mfm * vol
    sum_mfv = mfv.rolling(window=window, min_periods=window).sum()
    sum_vol = vol.rolling(window=window, min_periods=window).sum()

    cmf = sum_mfv / sum_vol
    cmf.name = f"CMF_{window}"

    return cmf