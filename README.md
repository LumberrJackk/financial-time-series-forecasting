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
- Financial time series treated as a signal
- Short-Time Fourier Transform (STFT) applied to generate spectrograms
- CNN model trained on spectrograms to predict future stock prices

## Stocks Used
- TCS (Tata Consultancy Services) — TCS.NS
- Infosys — INFY.NS
- Reliance Industries — RELIANCE.NS

## Project Structure
```
├── main.py                  # Data collection and time series plot
├── signal_processing.py     # Fourier Transform and STFT spectrogram
├── cnn_model.py             # CNN model training and prediction
├── analysis.py              # Evaluation and error analysis
```

## Results
| Metric | Value |
|--------|-------|
| MSE | 21,165.37 |
| RMSE | 145.48 INR |
| MAE | 111.32 INR |
| Avg Actual Price | ₹2,805.25 |
| Avg Predicted Price | ₹2,795.17 |

## Requirements
```
numpy
pandas
matplotlib
yfinance
scipy
scikit-learn
tensorflow
seaborn
```

## How to Run
```bash
pip install numpy pandas matplotlib yfinance scipy scikit-learn tensorflow seaborn
python main.py
python signal_processing.py
python cnn_model.py
python analysis.py
```

## References
1. Y. Zhang and C. Aggarwal, "Stock Market Prediction Using Deep Learning," IEEE Access.
2. A. Tsantekidis et al., "Deep Learning for Financial Time Series Forecasting."
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
4. A. Borovykh et al., "Conditional Time Series Forecasting with CNNs."
