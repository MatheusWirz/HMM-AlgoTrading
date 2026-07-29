from indicators import compute_cmf, compute_rsi, compute_obv, markovmodel
import pandas as pd
import numpy as np
import itertools


#==============INPUTS================#
rsi_max_window = 60
cmf_max_window = 60
states_max = 6
train_max = 0.50

data_type = ['intraday', 'daily']
assets = ['AAPL',
          'AMZN']



#==============BACKTEST================#
for asset, dtype in itertools.product(assets, data_type):
    # import datas
    df = pd.read_csv(f'datas/{asset}_{dtype}.csv')

    # Parameters
    risk_free = 0.05
    wRSI = range(3, rsi_max_window+1)
    wCMF = range(3, cmf_max_window+1)
    n_states = range(3, states_max+1)
    data_to_train = np.arange(0.15, train_max+0.01, 0.05)


    #==============CALCULATE===============#
    # Out loop
    dOBV = compute_obv(df)
    df['dOBV'] = dOBV
    results_list = []

    # Permute parameters
    for wrsi, wcmf, nstates, train in itertools.product(wRSI, wCMF, n_states, data_to_train):

        # Print
        train_cent = train*100
        model = f'{int(wrsi)}-{int(wcmf)}-{int(nstates)}-{int(train_cent)}'
        print(f'Training model: {model}...')

        # Indicators
        rsi = compute_rsi(df, wrsi)
        cmf = compute_cmf(df, wcmf)
        df['RSI'] = rsi
        df['CMF'] = cmf

        # Run markov
        result = markovmodel(df, train, nstates)
        print('   Model has been trained')

        # Export result of model to CSV
        result.to_csv(f'results/{model}_{asset}_{dtype}.csv')
        print('   Exported to CSV\n')

        # Returns, Vol, Sharpe
        returns = result['Strategy_cumulated_return'].iloc[-1]
        vol = result['Strategy_returns'].std()
        sharpe = (returns - risk_free) / vol

        # ==============ADD-TO-DATAFRAME===============#
        results_row = {
            'asset': asset,
            'model': model,
            'datatype': dtype,
            'sharpe': sharpe,
            'returns': returns,
            'vol': vol
        }
        results_list.append(results_row)

# Results Dataframe
df_results = pd.DataFrame(results_list)
df_results.sort_values(by='sharpe', ascending=False, inplace=True)

# ==============EXPORT-FINAL-RESULT===============#
print('\n\nAll models has been trained!')
max_model = f'{int(rsi_max_window)}-{int(cmf_max_window)}-{int(states_max)}-{int(train_max*100)}'
df_results.to_excel(f'results/MODELS_TRAINED_{max_model}.xlsx', index=False)