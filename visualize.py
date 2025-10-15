import matplotlib.pyplot as plt

def plot_predictions(y_test, preds_dict, out_path):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual")
    for name, preds in preds_dict.items():
        plt.plot(preds, label=f"Predicted - {name}")
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.savefig(out_path)
    plt.close()