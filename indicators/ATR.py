import pandas as pd
import numpy as np

# ATR (Average True Range) [Indicador de Volatilidade]

def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    # Calcula o close de d-1
    prev_close = close.shift(1)

    # Calcula os 3 componentes do True Range
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    # O True Range é o valor máximo dos 3
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Agora o ATR
    atr = tr.ewm(com=window - 1, min_periods=window).mean()

    # Normaliza o ATR
    atrn = (atr / close) * 100

    atrn.name = f'ATRN_{window}'

    return atrn