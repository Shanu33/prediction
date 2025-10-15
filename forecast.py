import pandas as pd
import numpy as np

def predict_next(model, scaler, history_df):
    df_feat = history_df.copy()
    df_feat['Return'] = df_feat['Close'].pct_change()
    df_feat['Lag1'] = df_feat['Close'].shift(1)
    df_feat['Lag2'] = df_feat['Close'].shift(2)
    df_feat['Lag3'] = df_feat['Close'].shift(3)
    df_feat['Lag4'] = df_feat['Close'].shift(4)
    df_feat['Lag5'] = df_feat['Close'].shift(5)
    df_feat = df_feat.dropna()

    if df_feat.empty:
        # Agar enough rows nahi hain, last close ko repeat kar do
        last_close = history_df['Close'].iloc[-1]
        return last_close

    X = df_feat[['Close', 'Volume', 'Return', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5']].values[-1].reshape(1, -1)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]


def forecast_future(model, scaler, df, steps=10):
    history = df.copy()
    preds = []
    for _ in range(steps):
        next_val = predict_next(model, scaler, history)
        preds.append(next_val)
        # append predicted row
        last_volume = history['Volume'].iloc[-1] if 'Volume' in history.columns else 0
        history = pd.concat(
            [history, pd.DataFrame({"Close": [next_val], "Volume": [last_volume]})],
            ignore_index=True
        )
    return preds
