import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load stock data
prices = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)
signal = prices['TCS.NS'].values

# Normalize
scaler = MinMaxScaler()
signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

print("Signal length:", len(signal_norm))

# Generate spectrogram windows for CNN input
window_length = 64
hop_size = 32
future_days = 5   # predict price 5 days ahead

X = []  # spectrogram inputs
y = []  # future price targets

# Slide a window across the signal
for i in range(0, len(signal_norm) - window_length - future_days, hop_size):
    segment = signal_norm[i : i + window_length]

    # Compute STFT for this segment
    f, t, Zxx = stft(segment, nperseg=16, noverlap=8)
    spec = np.abs(Zxx)

    X.append(spec)
    y.append(signal_norm[i + window_length + future_days])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Add channel dimension for CNN
X = X[..., np.newaxis]
print("X shape after channel dim:", X.shape)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

import tensorflow as tf
from tensorflow.keras import layers, models

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1)  # output: predicted price
])

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

print("\nTraining complete!")

# Save the model
model.save('stock_cnn_model.keras')
print("Model saved!")

# Make predictions on test set
y_pred = model.predict(X_test).flatten()

# Convert back to original price scale
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Calculate MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_actual, y_pred_actual)
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"Root MSE: {mse**0.5:.2f} INR")

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, marker='o', label='Actual Price', color='blue')
plt.plot(y_pred_actual, marker='s', label='Predicted Price', color='red', linestyle='--')
plt.title('CNN Prediction vs Actual TCS Stock Price')
plt.xlabel('Sample')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.savefig('plot_predictions.png')
plt.show()
print("Predictions plot saved!")

# Plot training loss curve
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.savefig('plot_loss_curve.png')
plt.show()
print("Loss curve plot saved!")
