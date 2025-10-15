import os
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data import load_data, create_features
from model import fit_scaler, scale_with, train_gb, train_mlp, save_model
from evaluate import evaluate_model
from visualize import plot_predictions

os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

def run_demo():
    df = load_data(config.SYMBOL, config.START_DATE, config.END_DATE)
    print("Data sample:\n", df.head())

    X, y = create_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, shuffle=False
    )

    scaler, X_train_scaled = fit_scaler(X_train)
    X_test_scaled = scale_with(scaler, X_test)

    gb = train_gb(X_train_scaled, y_train)
    mlp = train_mlp(X_train_scaled, y_train)

    preds_gb, rmse_gb, mape_gb = evaluate_model(gb, X_test_scaled, y_test)
    preds_mlp, rmse_mlp, mape_mlp = evaluate_model(mlp, X_test_scaled, y_test)

    print(f"GB -> RMSE: {rmse_gb:.2f}, MAPE: {mape_gb:.2%}")
    print(f"MLP -> RMSE: {rmse_mlp:.2f}, MAPE: {mape_mlp:.2%}")

    preds_df = pd.DataFrame({
        "Actual": y_test.values,
        "GB_Pred": preds_gb,
        "MLP_Pred": preds_mlp
    }, index=y_test.index)
    preds_df.to_csv(os.path.join(config.ARTIFACTS_DIR, "test_results.csv"))

    plot_predictions(y_test, {"GB": preds_gb, "MLP": preds_mlp},
                     os.path.join(config.ARTIFACTS_DIR, "predictions.png"))

    save_model(gb, os.path.join(config.ARTIFACTS_DIR, "gb_model.joblib"))
    save_model(mlp, os.path.join(config.ARTIFACTS_DIR, "mlp_model.joblib"))

if __name__ == "__main__":
    run_demo()
