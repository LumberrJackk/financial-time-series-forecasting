import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Load data
prices = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)

# Normalize
scaler = MinMaxScaler()
signal = prices['TCS.NS'].values
signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

# Rebuild X and y (same as cnn_model.py)
window_length = 64
hop_size = 32
future_days = 5

X, y = [], []
for i in range(0, len(signal_norm) - window_length - future_days, hop_size):
    segment = signal_norm[i : i + window_length]
    f, t, Zxx = stft(segment, nperseg=16, noverlap=8)
    X.append(np.abs(Zxx))
    y.append(signal_norm[i + window_length + future_days])

X = np.array(X)[..., np.newaxis]
y = np.array(y)

# Load saved model
model = tf.keras.models.load_model('stock_cnn_model.keras')

# Predict on all samples
y_pred_norm = model.predict(X).flatten()

# Convert back to INR
y_actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
y_pred   = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

# Metrics
mse  = mean_squared_error(y_actual, y_pred)
rmse = mse ** 0.5
mae  = np.mean(np.abs(y_actual - y_pred))

print("=" * 40)
print("       ANALYSIS RESULTS")
print("=" * 40)
print(f"MSE  (Mean Squared Error) : {mse:.2f}")
print(f"RMSE (Root MSE)           : {rmse:.2f} INR")
print(f"MAE  (Mean Abs Error)     : {mae:.2f} INR")
print(f"Avg actual price          : {y_actual.mean():.2f} INR")
print(f"Avg predicted price       : {y_pred.mean():.2f} INR")
print("=" * 40)

# Plot all predictions vs actual
plt.figure(figsize=(12, 5))
plt.plot(y_actual, marker='o', label='Actual Price', color='blue')
plt.plot(y_pred,   marker='s', label='Predicted Price', color='red', linestyle='--')
plt.title('CNN: All Predictions vs Actual TCS Stock Price')
plt.xlabel('Window Sample')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.savefig('plot_analysis.png')
print("Analysis plot saved!")

# Error per sample
errors = np.abs(y_actual - y_pred)
plt.figure(figsize=(12, 4))
plt.bar(range(len(errors)), errors, color='orange')
plt.title('Absolute Error per Sample')
plt.xlabel('Sample')
plt.ylabel('Error (INR)')
plt.tight_layout()
plt.savefig('plot_errors.png')
print("Error plot saved!")
