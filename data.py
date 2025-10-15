import yfinance as yf
import pandas as pd

def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)

    if 'Adj Close' in df.columns:
        df = df[['Adj Close']].rename(columns={'Adj Close': 'Close'})
    else:
        df = df[['Close']]

    return df

def create_features(df, lags=[1, 5, 10]):
    df_feat = df.copy()
    df_feat['Return'] = df_feat['Close'].pct_change()
    df_feat['MA10'] = df_feat['Close'].rolling(10).mean()
    df_feat['MA20'] = df_feat['Close'].rolling(20).mean()
    df_feat['Volatility'] = df_feat['Return'].rolling(10).std()

    for lag in lags:
        df_feat[f'Lag_{lag}'] = df_feat['Return'].shift(lag)

    df_feat = df_feat.dropna()
    X = df_feat.drop(columns=['Close'])
    y = df_feat['Close']

    y = y.squeeze()
    return X, y