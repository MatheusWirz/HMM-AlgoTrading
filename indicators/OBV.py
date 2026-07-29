import pandas as pd
import numpy as np

# OBV (On Balance Volume)

def compute_obv(df: pd.DataFrame) -> pd.Series:

    close = df['close'].astype(float)
    volume = df['volume'].astype(float)

    # Compara o close de d com d-1
    direction = np.sign(close.diff(1).fillna(0)) # np.sign() retorna o '1' se for positivo, '-1' se for negativo

    # Calcula volume e OBV
    signed_volume = direction * volume
    obv = signed_volume.cumsum()
    obv.name = "OBV"

    # dOBV
    pct_obv = obv.pct_change().fillna(0)
    pct_close = close.pct_change().fillna(0)
    dOBV = pct_close - pct_obv

    return dOBV