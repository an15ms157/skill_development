# This script loads index data from JSON, calculates SMA, and saves the result as a new JSON file.
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

input_json = './data/SPX_data.json'  # Path to the downloaded data
output_json = './data/SPX_sma_data.json'  # Path to save SMA data
sma_window1 = 200  # First SMA window
sma_window2 = 600  # Second SMA window

# Load data
with open(input_json, 'r') as f:
    records = json.load(f)

df = pd.DataFrame(records)

# Calculate SMAs on 'Close' column
if 'Close' in df.columns:
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Ensure 'Close' is numeric
    df['Close'].fillna(method='ffill', inplace=True)  # Forward-fill NaN values
    df['SMA_1'] = df['Close'].rolling(window=sma_window1).mean()
    df['SMA_2'] = df['Close'].rolling(window=sma_window2).mean()
    # Drop rows where either SMA_1 or SMA_2 is NaN (i.e., before their respective windows)
    min_window = max(sma_window1, sma_window2)
    df = df[df.index >= min_window - 1].reset_index(drop=True)
else:
    raise ValueError("'Close' column not found in data.")

# Save with SMAs
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
result = df.to_dict(orient='records')
with open(output_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved SMA data with windows {sma_window1} and {sma_window2} to {output_json}")

# Plot using the JSON file just created
json_path = './data/SPX_sma_data.json'
with open(json_path, 'r') as f:
    records = json.load(f)
df = pd.DataFrame(records)
df['Date'] = pd.to_datetime(df['Date'])

plt.style.use('default')
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='black')
plt.plot(df['Date'], df['SMA_1'], label=f'SMA1 (window={sma_window1})', color='blue')
plt.plot(df['Date'], df['SMA_2'], label=f'SMA2 (window={sma_window2})', color='red')
plt.title(f'SPX Close Price with SMA1 (window={sma_window1}) and SMA2 (window={sma_window2}) (from JSON)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Ensure the output directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), 'plots/sma'), exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plots/sma/SPX_sma_plot_from_json.png'))
plt.show()
