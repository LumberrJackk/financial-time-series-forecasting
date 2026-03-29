import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

prices = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)

tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
names   = ['TCS', 'Infosys', 'Reliance']

window_length = 64
hop_size      = 32
future_days   = 5

results = {}

for ticker, name in zip(tickers, names):
    print(f"\nProcessing {name}...")

    signal = prices[ticker].values
    scaler = MinMaxScaler()
    signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(0, len(signal_norm) - window_length - future_days, hop_size):
        segment = signal_norm[i : i + window_length]
        f, t, Zxx = stft(segment, nperseg=16, noverlap=8)
        X.append(np.abs(Zxx))
        y.append(signal_norm[i + window_length + future_days])

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same',
            input_shape=X_train.shape[1:]),
        tf.keras.layers.MaxPooling2D((2,2), padding='same'),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8,
              validation_data=(X_test, y_test), verbose=0)

    y_pred = model.predict(X_test).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred   = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()

    mse  = mean_squared_error(y_actual, y_pred)
    rmse = mse ** 0.5
    results[name] = {'actual': y_actual, 'pred': y_pred, 'mse': mse, 'rmse': rmse}
    print(f"{name} — RMSE: {rmse:.2f} INR")

# Plot all 3 predictions
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
colors = ['blue', 'green', 'red']

for i, name in enumerate(names):
    actual = results[name]['actual']
    pred   = results[name]['pred']
    rmse   = results[name]['rmse']
    axes[i].plot(actual, marker='o', label='Actual', color=colors[i])
    axes[i].plot(pred, marker='s', label='Predicted',
                 color=colors[i], linestyle='--', alpha=0.7)
    axes[i].set_title(f'{name} — Predicted vs Actual (RMSE: {rmse:.2f} INR)')
    axes[i].set_xlabel('Sample')
    axes[i].set_ylabel('Price (INR)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_predictions_all.png', dpi=150)
print("\nAll predictions plot saved!")

# Print summary table
print("\n" + "="*45)
print(f"{'Company':<12} {'MSE':>12} {'RMSE':>12}")
print("="*45)
for name in names:
    print(f"{name:<12} {results[name]['mse']:>12.2f} {results[name]['rmse']:>10.2f} INR")
print("="*45)
