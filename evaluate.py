from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)
    return preds, rmse, mape