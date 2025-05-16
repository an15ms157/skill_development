# This script simulates a SIP (Systematic Investment Plan) where $1 is invested on a random day each month in the index.
# It computes and plots ROIC, net capital invested, and net valuation over time.
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

TICKER = 'SPX'
DATA_PATH = f'data/{TICKER}_data.json'
fee_rate = 0.03  # maintanance fee of 1% per year

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file for ticker {TICKER} not found at {DATA_PATH}")

with open(DATA_PATH, 'r') as f:
    records = json.load(f)

df = pd.DataFrame(records)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Group by year and month for SIP
df['YearMonth'] = df['Date'].dt.to_period('M')
months = df['YearMonth'].unique()

np.random.seed(42)  # For reproducibility

invested_dates = []
invested_prices = []
invested_capital = []
units_held = []
net_valuation = []
roic = []
total_invested = 0
units = 0

for month in months:
    month_df = df[df['YearMonth'] == month]
    if month_df.empty:
        continue
    # Pick a random day in the month
    idx = np.random.choice(month_df.index)
    price = month_df.loc[idx, 'Close']
    date = month_df.loc[idx, 'Date']
    # Buy $1 worth of index
    units += 1 / price
    total_invested += 1
    invested_dates.append(date)
    invested_prices.append(price)
    invested_capital.append(total_invested)
    units_held.append(units)
    # Net valuation at this point (using latest close in month)
    latest_price = month_df['Close'].iloc[-1]
    value = units * latest_price
    # Apply yearly maintenance fee at the end of each year
    if len(invested_dates) > 1:
        prev_year = invested_dates[-2].year
        curr_year = invested_dates[-1].year
        if curr_year != prev_year:
            # Deduct maintenance fee (e.g., 1%) from value
            value *= (1 - fee_rate)
    net_valuation.append(value)
    roic.append((value - total_invested) / total_invested)

# Plotting
plt.figure(figsize=(12, 6))
ax1 = plt.gca()

# Plot ROIC on left y-axis
ax1.plot(invested_dates, roic, label='ROIC', color='purple')
ax1.set_ylabel('ROIC')
ax1.legend(loc='upper left')
ax1.grid(True)

# Plot Net Capital Invested and Net Valuation on right y-axis
ax2 = ax1.twinx()
ax2.plot(invested_dates, invested_capital, label='Net Capital Invested', color='blue')
ax2.plot(invested_dates, net_valuation, label='Net Valuation', color='green')
ax2.set_ylabel('Dollars ($)')
ax2.legend(loc='upper right')

plt.title(f'{TICKER} SIP: Capital Invested, Net Valuation, and ROIC Over Time')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig(f'plots/strategy/{TICKER}_sip_simulation.png')
plt.show()

# Save data to JSON based on time
output_data = []
for date, capital, valuation, r in zip(invested_dates, invested_capital, net_valuation, roic):
    output_data.append({
        'Date': date.strftime('%Y-%m-%d'),
        'NetCapitalInvested': capital,
        'NetValuation': valuation,
        'ROIC': r
    })

# Print all data before saving to file
print("Date".ljust(12), "NetCapitalInvested".rjust(18), "NetValuation".rjust(15), "ROIC".rjust(10))
for row in output_data:
    print(
        row['Date'].ljust(12),
        f"{row['NetCapitalInvested']:18.2f}",
        f"{row['NetValuation']:15.2f}",
        f"{row['ROIC']:10.4f}"
    )

with open(f'data/{TICKER}_sipNoExit_simulation_timeseries.json', 'w') as f:
    json.dump(output_data, f, indent=2)
