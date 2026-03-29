import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download stock data for 3 Indian companies
tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']

data = {}
for ticker in tickers:
    df = yf.download(ticker, start='2020-01-01', end='2024-01-01', auto_adjust=True)
    data[ticker] = df['Close'].squeeze()
    print(f"Downloaded {ticker}: {len(df)} rows")

# Combine into one dataframe
prices = pd.DataFrame(data)
prices.dropna(inplace=True)

print("\nFirst 5 rows:")
print(prices.head())

print("\nShape:", prices.shape)

# Save to CSV
prices.to_csv('stock_prices.csv')
print("\nSaved to stock_prices.csv")

# Plot the time series
plt.figure(figsize=(12, 5))
for col in prices.columns:
    plt.plot(prices.index, prices[col], label=col)

plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.savefig('plot_timeseries.png')
plt.show()
print("Time series plot saved!")
