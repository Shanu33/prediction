import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import yfinance as yf
from datetime import timedelta
import plotly.graph_objects as go

from data import create_features
from evaluate import evaluate_model
from forecast import forecast_future

st.title("📈 Stock Prediction & Forecasting Demo")

# --- User Inputs ---
symbol = st.text_input("Enter stock symbol", "AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2023-01-01"))

timeframe = st.radio("Select timeframe:", ("Daily", "Hourly"))
interval = "1d" if timeframe == "Daily" else "1h"

mode = st.radio("Visualization mode:",
                ("Actual Prices Only", "Predicted vs Actual", "Future Forecast"))

selected_models = st.multiselect(
    "Select model(s)",
    options=["GB", "MLP"],
    default=["GB", "MLP"]
)

# --- Load Stock Data ---
@st.cache_data
def load_stock_data(symbol, start, end, interval):
    if interval == "1h":
        max_days = 730
        if (end - start).days > max_days:
            start = end - timedelta(days=max_days)
            st.warning("⚠️ Hourly data is limited to the last 2 years. Adjusted date range automatically.")

    df = yf.download(symbol, start=start, end=end, interval=interval)

    if df.empty:
        return pd.DataFrame()

    # Keep Close + Volume
    keep_cols = []
    if 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})
        keep_cols = ['Close']
    elif 'Close' in df.columns:
        keep_cols = ['Close']

    if 'Volume' in df.columns:
        keep_cols.append('Volume')

    df = df[keep_cols]
    return df

df = load_stock_data(symbol, start_date, end_date, interval)

if df.empty or len(df) < 100:
    st.error("⚠️ Not enough stock data fetched! Try changing the date range or use 'Daily' interval.")
    st.stop()

# --- Feature Engineering ---
X, y = create_features(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train Models ---
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_scaled, y)

mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=42, early_stopping=True)
mlp.fit(X_scaled, y)

# --- Evaluate Models ---
preds_gb, rmse_gb, mape_gb = evaluate_model(gb, X_scaled, y)
preds_mlp, rmse_mlp, mape_mlp = evaluate_model(mlp, X_scaled, y)

st.subheader("📊 Model Performance")
st.write(f"**GB** → RMSE: {rmse_gb:.2f}, MAPE: {mape_gb:.2%}")
st.write(f"**MLP** → RMSE: {rmse_mlp:.2f}, MAPE: {mape_mlp:.2%}")

# --- Plotly Interactive Graph ---
fig = go.Figure()

# Actual Prices
fig.add_trace(go.Scatter(
    x=y.index,
    y=y.values,
    mode='lines+markers',
    name='Actual',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
))

if mode == "Predicted vs Actual":
    if "GB" in selected_models:
        fig.add_trace(go.Scatter(
            x=y.index,
            y=preds_gb,
            mode='lines+markers',
            name='GB Prediction',
            line=dict(color='orange'),
            hovertemplate='Date: %{x}<br>GB Predicted: $%{y:.2f}<extra></extra>'
        ))
    if "MLP" in selected_models:
        fig.add_trace(go.Scatter(
            x=y.index,
            y=preds_mlp,
            mode='lines+markers',
            name='MLP Prediction',
            line=dict(color='green'),
            hovertemplate='Date: %{x}<br>MLP Predicted: $%{y:.2f}<extra></extra>'
        ))

elif mode == "Future Forecast":
    steps_ahead = st.number_input("Steps to forecast ahead:", min_value=1, max_value=200, value=20)
    future_index = pd.date_range(start=df.index[-1], periods=steps_ahead+1, freq=("D" if timeframe=="Daily" else "H"))[1:]

    if "GB" in selected_models:
        future_preds_gb = forecast_future(gb, scaler, df, steps=steps_ahead)
        fig.add_trace(go.Scatter(
            x=future_index,
            y=future_preds_gb,
            mode='lines+markers',
            name='GB Forecast',
            line=dict(color='orange'),
            hovertemplate='Date: %{x}<br>GB Forecast: $%{y:.2f}<extra></extra>'
        ))
        st.write("📌 GB Future Forecast:", pd.DataFrame({"Date": future_index, "Price (USD)": future_preds_gb}))

    if "MLP" in selected_models:
        future_preds_mlp = forecast_future(mlp, scaler, df, steps=steps_ahead)
        fig.add_trace(go.Scatter(
            x=future_index,
            y=future_preds_mlp,
            mode='lines+markers',
            name='MLP Forecast',
            line=dict(color='green'),
            hovertemplate='Date: %{x}<br>MLP Forecast: $%{y:.2f}<extra></extra>'
        ))
        st.write("📌 MLP Future Forecast:", pd.DataFrame({"Date": future_index, "Price (USD)": future_preds_mlp}))

# Layout
fig.update_layout(
    title='📈 Stock Prices & Predictions',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)
