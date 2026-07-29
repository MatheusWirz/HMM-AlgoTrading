import pandas as pd
import numpy as np

# ADX (Average Directional Index)

def compute_adx(df: pd.DataFrame, window: int) -> pd.Series:

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    # As próximas linhas são apenas para calcular o ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.ewm(com=window - 1, min_periods=window).mean()
    # ~ #

    # Calcula o movimento direcional
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = low - prev_low

    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0)

    # Faz um Exponential Moving Average para suavizar ruídos
    dmp_ema = dm_plus.ewm(com=window - 1, min_periods=window).mean()
    dmn_ema = dm_minus.ewm(com=window - 1, min_periods=window).mean()

    # Calcula o Indice Direcional (DI+ e DI-)
    # Adiciona um número mínimo para evitar divisão por 0
    di_plus = (dmp_ema / (atr + 1e-9)) * 100
    di_minus = (dmn_ema / (atr + 1e-9)) * 100

    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100

    adx = dx.ewm(com=window -1, min_periods=window).mean()

    adx.name = f'ADX_{window}'

    return adx