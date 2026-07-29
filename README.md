# Algorithmic Trading with Hidden Markov Models (HMM)

Este repositório contém a implementação de um modelo quantitativo de *Algorithmic Trading* baseado em Modelos Ocultos de Markov (Hidden Markov Models) para a identificação de transições de regimes de mercado.

## Descrição do Projeto

O núcleo desta estratégia reside na utilização de um modelo estatístico não supervisionado (`GaussianHMM`) para prever os estados latentes do mercado financeiro. Através da análise combinada de indicadores técnicos de momento e fluxo, o modelo classifica os períodos do mercado em diferentes regimes e gera sinais sistemáticos de compra, venda ou encerramento de posições com base nas características estatísticas de cada estado.

## Indicadores Técnicos Implementados

A biblioteca conta com a implementação nativa e vetorizada (via `pandas` e `numpy`) de diversos indicadores técnicos utilizados para alimentar o modelo e capturar diferentes dimensões da dinâmica de preços:

*   **RSI (Relative Strength Index):** Mede a magnitude das mudanças recentes de preço. É utilizado no modelo como o critério principal para a definição lógica dos estados ocultos (sobrecompra vs. sobrevenda).
*   **CMF (Chaikin Money Flow):** Combina dados de preço e volume para medir a pressão de compra e venda do mercado ao longo de uma janela de tempo.
*   **OBV & dOBV (On Balance Volume):** Mede o fluxo cumulativo de volume. O modelo estende o OBV tradicional utilizando o `dOBV`, uma métrica customizada que calcula a divergência entre a variação percentual do preço de fechamento e a variação percentual do OBV.
*   **ATR (Average True Range):** Indicador clássico de volatilidade, implementado em sua versão normalizada (`ATRN`) em relação ao preço de fechamento.
*   **ADX (Average Directional Index):** Quantifica a força de uma tendência utilizando médias móveis exponenciais (EMA) do movimento direcional.

## Modelagem Estocástica e Pipeline Quantitativo

O pipeline de modelagem da classe `markovmodel` segue rigorosamente as seguintes etapas:

1.  **Seleção de Features:** Utiliza a matriz tridimensional composta por `RSI`, `CMF` e `dOBV`.
2.  **Pré-processamento:** Realiza a separação *out-of-sample* (treino e teste) e a padronização dos dados (Z-score) utilizando o `StandardScaler`.
3.  **Otimização do Modelo:** Para contornar mínimos locais, o sistema treina múltiplas instâncias do `GaussianHMM` com diferentes sementes aleatórias (padrão de 30 tentativas) e seleciona o modelo que apresentar a maior Verossimilhança (*Log-Likelihood score*).
4.  **Classificação de Regimes:** Os estados latentes são traduzidos em posições operacionais através da análise dos centróides (médias) das distribuições. O regime com o menor RSI médio é interpretado como subprecificado (sinal de Compra/Long), o de maior RSI como sobreprecificado (sinal de Venda/Short), e os regimes intermediários como transição (Encerramento de posição).
5.  **Backtesting Dinâmico:** Utiliza o algoritmo de *Viterbi* para decodificar os estados mais prováveis na base de teste e computa o retorno cumulativo da estratégia em comparação ao retorno *buy-and-hold* do ativo subjacente.

## Métricas de Avaliação

O módulo de estatísticas integrado (`metricas_simples`) fornece as principais métricas de desempenho ajustadas ao risco para avaliação do backtest [cite: 1, 2]:
*   **CAGR** (Taxa de Crescimento Anual Composta)
*   **Volatilidade Anualizada** 
*   **Índice Sharpe (Aproximado)**

## Requisitos e Dependências

O projeto utiliza um stack tradicional de *Data Science* e *Machine Learning* em Python:
*   `pandas`
*   `numpy`
*   `scikit-learn` (`StandardScaler`)
*   `hmmlearn` (`GaussianHMM`)
