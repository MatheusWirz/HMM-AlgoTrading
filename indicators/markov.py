import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def markovmodel(df: pd.DataFrame, data_to_train: float, n_states = 3, n_iter=1000, n_tentativas = 30) -> pd.DataFrame:
    # Matrix indicators
    features = ['RSI', 'CMF', 'dOBV']
    datas = df[features]

    # Split datas into training (train) and backtest (test)
    split_index = int(len(datas) * data_to_train)
    data_train = datas.iloc[:split_index].copy()
    data_test = datas.iloc[split_index:].copy()

    # Drop NaN
    data_train = data_train.dropna()

    # Errors
    if data_train.empty:
        print('Erro: Empty dataFrame')
        return

    # Padronizar os dados (z-score)
    scaler = StandardScaler()
    datas_standard = scaler.fit_transform(data_train)

    # Best model
    score = -np.inf
    model = None

    for i in range(n_tentativas):
        model_i = GaussianHMM(n_components=n_states,
                            covariance_type="full",
                            n_iter=n_iter,
                            random_state=i)
        try:
            # Trains better model
            model_i.fit(datas_standard)

            # Calculate Likelihood (score)
            score_i = model_i.score(datas_standard)
            if score_i > score:
                score = score_i
                model = model_i

        except ValueError as e:
            print(f'Tentativa {i+1}/{n_tentativas} falhou: {e}')
            continue

    # States - Hidden Markov
    states = pd.DataFrame(model.means_, columns=features)
    buy = states['RSI'].idxmin()
    sell = states['RSI'].idxmax()
    close = [s for s in range(n_states) if s not in [buy, sell]]

    # Viterbi - Calculate states
    data_test_std = scaler.transform(data_test[features])
    estimate_states = model.predict(data_test_std)
    data_test['State'] = estimate_states

    # Positions
    position = pd.Series(index=data_test.index, dtype=int)
    position[data_test['State'] == buy] = 1
    position[data_test['State'] == sell] = -1
    position[data_test['State'].isin(close)] = 0

    # Strategy return
    data_test['Position'] = position.shift(1).fillna(0)
    data_test['close'] = df['close'].reindex(data_test.index)
    asset_return = data_test['close'].pct_change()
    data_test['Strategy_returns'] = asset_return * data_test['Position']
    data_test['Strategy_cumulated_return'] = (1 + data_test['Strategy_returns']).cumprod() - 1

    return data_test