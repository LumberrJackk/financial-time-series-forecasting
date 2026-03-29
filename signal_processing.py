import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

# Load the saved stock data
prices = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)

print("Loaded data shape:", prices.shape)
print(prices.head())

# We will work with TCS stock price as our signal
signal = prices['TCS.NS'].values
print("\nSignal length:", len(signal))

# Normalize the signal between 0 and 1
signal_min = signal.min()
signal_max = signal.max()
signal_norm = (signal - signal_min) / (signal_max - signal_min)

print("Signal normalized. Min:", signal_norm.min(), "Max:", signal_norm.max())

# Plot the normalized signal
plt.figure(figsize=(12, 4))
plt.plot(signal_norm)
plt.title('Normalized TCS Stock Price Signal')
plt.xlabel('Time (days)')
plt.ylabel('Normalized Price')
plt.tight_layout()
plt.savefig('plot_normalized_signal.png')
plt.show()
print("Normalized signal plot saved!")

# Apply Fourier Transform
fft_vals = np.fft.fft(signal_norm)
fft_magnitude = np.abs(fft_vals)
frequencies = np.fft.fftfreq(len(signal_norm))

# Only take the positive frequencies (first half)
half = len(frequencies) // 2
freq_pos = frequencies[:half]
magnitude_pos = fft_magnitude[:half]

# Plot frequency spectrum
plt.figure(figsize=(12, 4))
plt.plot(freq_pos, magnitude_pos)
plt.title('Frequency Spectrum of TCS Stock Price')
plt.xlabel('Frequency (cycles per day)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig('plot_frequency_spectrum.png')
plt.show()
print("Frequency spectrum plot saved!")

# Apply STFT to generate spectrogram
window_length = 64   # each window is 64 days
hop_size = 32        # window moves 32 days at a time

f, t, Zxx = stft(signal_norm, nperseg=window_length, noverlap=window_length - hop_size)

# Get the magnitude (spectrogram)
spectrogram = np.abs(Zxx)

print("Spectrogram shape:", spectrogram.shape)

# Plot the spectrogram
plt.figure(figsize=(12, 5))
plt.pcolormesh(t, f, spectrogram, shading='gouraud', cmap='inferno')
plt.title('STFT Spectrogram of TCS Stock Price')
plt.xlabel('Time (days)')
plt.ylabel('Frequency')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.savefig('plot_spectrogram.png')
plt.show()
print("Spectrogram plot saved!")
