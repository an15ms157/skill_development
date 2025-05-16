import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_and_store_price_logs(ticker):
    # Fetch historical market data from yfinance
    stock_data = yf.Ticker(ticker).history(period="max")

    # Plot the closing price
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label=f'{ticker} Closing Price')
    plt.title(f'{ticker} Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    # Ensure the log directory exists
    log_dir = os.path.join(os.path.dirname(__file__), '../log')
    os.makedirs(log_dir, exist_ok=True)

    # Save the price logs to a CSV file with adequate spacing for columnar appearance
    csv_file_path = os.path.join(log_dir, f'{ticker}_price_log.csv')
    stock_data.to_csv(csv_file_path, index=True, sep=',', float_format='%.6f')
    print(f"Price logs for {ticker} have been saved to {csv_file_path}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python valuation.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1]
    plot_and_store_price_logs(ticker)