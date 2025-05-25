# This script computes a 2x leveraged price index from historical data and plots both the original and 2x index.
# The 2x leveraged index is saved as a JSON file in ./data/ticker_2x.json.
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

TICKER = 'SPX'
LEVERAGE = 2
DATA_PATH = f'data/{TICKER}_data.json'
OUTPUT_JSON = f'data/{TICKER}_{LEVERAGE}x.json'

N = 600 
N2 = N // 2

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file for ticker {TICKER} not found at {DATA_PATH}")

with open(DATA_PATH, 'r') as f:
    records = json.load(f)

df = pd.DataFrame(records)
df['Date'] = pd.to_datetime(df['Date'])

returns = df['Close'].pct_change()
returns = returns.fillna(0)
lev_returns = returns * LEVERAGE
lev_index = [df['Close'].iloc[0]]
for r in lev_returns[1:]:
    lev_index.append(lev_index[-1] * (1 + r))
df[f'Close_xlev'] = lev_index


sma_N = pd.Series(lev_index).rolling(window=N).mean()
sma_N2 = pd.Series(lev_index).rolling(window=N2).mean()
df[f'SMA_2'] = sma_N
df[f'SMA_1'] = sma_N2

# Save the leveraged index and SMAs as JSON
out_records = df[['Date', f'Close_xlev', f'SMA_1', f'SMA_2']].copy()
out_records['Date'] = out_records['Date'].dt.strftime('%Y-%m-%d')
out_json = out_records.to_dict(orient='records')
os.makedirs('data', exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(out_json, f, indent=2)
print(f"Saved {len(out_json)} records to {OUTPUT_JSON}")

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label=f'{TICKER} Index')
plt.plot(df['Date'], df[f'Close_xlev'], label=f'{TICKER} {LEVERAGE}x Leveraged Index')
plt.plot(df['Date'], df[f'SMA_1'], label=f'{LEVERAGE}x {N}-day SMA', linestyle='--')
plt.plot(df['Date'], df[f'SMA_2'], label=f'{LEVERAGE}x {N2}-day SMA', linestyle=':')
plt.title(f'{TICKER} vs {TICKER} {LEVERAGE}x Leveraged Index with SMAs')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs(os.path.join(os.path.dirname(__file__), 'plots/lev'), exist_ok=True)
plt.savefig(os.path.join(os.path.dirname(__file__), f'plots/lev/{TICKER}_{LEVERAGE}x_plot.png'))
plt.show()
