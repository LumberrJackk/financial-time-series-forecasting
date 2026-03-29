# Pattern Recognition for Financial Time Series Forecasting

## Student Details
Name: **SIVANANDHA K**  
Registration Number: **TCR24CS064**  
Course: Pattern Recognition  
Assignment: Second Assignment  

---

## Objective
To explore how time-frequency signal processing and deep learning can be combined to predict stock prices using financial time series data.

## Method
- Financial time series treated as a non-stationary signal
- Short-Time Fourier Transform (STFT) applied to generate spectrograms
- CNN model trained on spectrograms to predict future stock prices

## Stocks Used
- TCS (Tata Consultancy Services) — TCS.NS
- Infosys — INFY.NS
- Reliance Industries — RELIANCE.NS
- Data period: January 2020 to January 2024 (992 trading days)

## Project Structure
```
├── main.py                    # Task 1 — Data collection and time series plot
├── signal_processing.py       # Task 2 — Fourier Transform and STFT spectrogram
├── cnn_model.py               # Task 3 — CNN model training and prediction
├── analysis.py                # Task 4 — Evaluation and error analysis
├── stock_prices.csv           # Downloaded stock price dataset
├── plot_timeseries.png        # Time series plot (Task 1)
├── plot_normalized_signal.png # Normalized signal plot
├── plot_frequency_spectrum.png# Frequency spectrum (Task 2)
├── plot_spectrogram.png       # STFT Spectrogram (Task 2)
├── plot_cnn_architecture.png  # CNN architecture diagram (Task 3)
├── plot_predictions.png       # Predicted vs actual prices (Task 3)
├── plot_loss_curve.png        # Training loss curve (Task 3)
├── plot_analysis.png          # Full prediction analysis (Task 4)
├── plot_errors.png            # Per-sample error plot (Task 4)
├── plot_frequency_all.png     # Frequency spectrum — all 3 companies
├── plot_spectrogram_all.png   # STFT Spectrogram — all 3 companies
└── plot_normalized_all.png    # Normalized signals — all 3 companies
```

## Results
| Metric | Value |
|--------|-------|
| MSE | 21,165.37 |
| RMSE | 145.48 INR |
| MAE | 111.32 INR |
| Avg Actual Price | ₹2,805.25 |
| Avg Predicted Price | ₹2,795.17 |
| Error Rate | ~5.2% |

## How to Run
```bash
# Install dependencies
pip install numpy pandas matplotlib yfinance scipy scikit-learn tensorflow seaborn

# Run in order
python main.py                 # Download data and plot time series
python signal_processing.py   # Generate spectrograms
python cnn_model.py           # Train CNN and predict
python analysis.py            # Evaluate results
python cnn_architecture.py    # Generate architecture diagram
```

## References
1. Y. Zhang and C. Aggarwal, "Stock Market Prediction Using Deep Learning," IEEE Access.
2. A. Tsantekidis et al., "Deep Learning for Financial Time Series Forecasting."
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
4. A. Borovykh et al., "Conditional Time Series Forecasting with CNNs."
