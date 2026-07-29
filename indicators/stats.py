import pandas as pd
import numpy as np


def metricas_simples(serie_retorno: pd.Series, freq_ano=252):
    """
    Métricas simples: CAGR, Vol anualizada, Sharpe ~ (média/vol)*sqrt(252).
    """
    ret = serie_retorno.fillna(0)
    cum = (1 + ret).cumprod()
    n = ret.shape[0]
    anos = n / freq_ano
    cagr = cum.iloc[-1] ** (1 / anos) - 1 if anos > 0 else np.nan
    vol_ann = ret.std() * np.sqrt(freq_ano)
    sharpe = ret.mean() / ret.std() * np.sqrt(freq_ano) if ret.std() > 0 else np.nan

    return {"CAGR": cagr, "Vol_Anualizada": vol_ann, "Sharpe_aprox": sharpe}