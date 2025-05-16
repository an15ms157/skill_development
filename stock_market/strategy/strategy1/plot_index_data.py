# This script loads index data from a JSON file (saved by ticker) and plots the historical closing prices.
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the ticker you want to plot
ticker = 'SPX'
json_path = f'./data/{ticker}_data.json'

if not os.path.exists(json_path):
    raise FileNotFoundError(f"Data file for ticker {ticker} not found at {json_path}")

with open(json_path, 'r') as f:
    records = json.load(f)

df = pd.DataFrame(records)

# Plot the closing price
df['Date'] = pd.to_datetime(df['Date'])
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label=f'{ticker} Close')
plt.title(f'{ticker} Historical Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)

# Ensure the output directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), 'plots/index'), exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), f'plots/index/{ticker}_close_plot.png'))
plt.show()
