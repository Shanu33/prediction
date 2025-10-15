import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def fit_scaler(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled

def scale_with(scaler, X):
    return scaler.transform(X)

def train_gb(X_train, y_train):
    gb = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)
    return gb

def train_mlp(X_train, y_train):
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)