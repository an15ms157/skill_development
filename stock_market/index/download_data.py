# Part 1: Download historical data (file 1)
# This script downloads index data from Yahoo Finance
import yfinance as yf
import pandas as pd
import os
from config import tickers, download_config  # Import tickers and download_config from config.py

def download_index_data():
    data = {}
    interval = download_config.get("interval", "1mo")  # Default to monthly if not specified
    start_date = download_config.get("start_date", "1970-01-01")  # Default to 1970-01-01 if not specified

    for country, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, interval=interval, progress=False)
        df = df[['Close']].rename(columns={'Close': country})
        print(f"Downloaded {country} index data up to: {df.index[-1]}")
        data[country] = df
    return data

if __name__ == "__main__":
    index_data = download_index_data()
    for country, df in index_data.items():
        output_folder = "index_data"
        os.makedirs(output_folder, exist_ok=True)
        df.to_csv(os.path.join(output_folder, f"{country}_index_data.csv"))