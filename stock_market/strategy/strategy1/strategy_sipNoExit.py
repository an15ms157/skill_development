# This script simulates a SIP (Systematic Investment Plan) where $1 is invested on a random day each month in the index.
# It computes and plots ROIC, net capital invested, and net valuation over time.
# python3 strategy_sipNoExit.py --start-date 2010-01-01 --end-date 2020-12-31
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

TICKER = 'SPX'
DATA_PATH = f'data/{TICKER}_data.json'
fee_rate = 0.00  # maintanance fee of 1% per year

# Define default start and end dates (can be overridden with command line arguments)
START_DATE = None  # Default is None, which means start from the beginning of the data
END_DATE = None    # Default is None, which means end at the latest date in the data

# Add command line argument support
import argparse
parser = argparse.ArgumentParser(description='SIP simulation with optional date range')
parser.add_argument('--start-date', type=str, help='Start date in YYYY-MM-DD format')
parser.add_argument('--end-date', type=str, help='End date in YYYY-MM-DD format')
args = parser.parse_args()

if args.start_date:
    START_DATE = pd.to_datetime(args.start_date)
if args.end_date:
    END_DATE = pd.to_datetime(args.end_date)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file for ticker {TICKER} not found at {DATA_PATH}")

with open(DATA_PATH, 'r') as f:
    records = json.load(f)

df = pd.DataFrame(records)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Filter data based on start and end dates if provided
if START_DATE:
    df = df[df['Date'] >= START_DATE]
    if df.empty:
        raise ValueError(f"No data available after start date {START_DATE}")
    
if END_DATE:
    df = df[df['Date'] <= END_DATE]
    if df.empty:
        raise ValueError(f"No data available before end date {END_DATE}")

# Group by year and month for SIP
df['YearMonth'] = df['Date'].dt.to_period('M')
months = df['YearMonth'].unique()

# Print date range being used
print(f"Running SIP simulation from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

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
ax2.set_ylabel('Dollars ($) - Log Scale')
#ax2.set_yscale('log')  # Set y-axis to logarithmic scale
ax2.legend(loc='upper right')

# Add date range to the title if custom dates were specified
title = f'{TICKER} SIP: Capital Invested, Net Valuation, and ROIC Over Time'
if START_DATE or END_DATE:
    date_range = []
    if START_DATE:
        date_range.append(START_DATE.strftime('%Y-%m-%d'))
    else:
        date_range.append(df['Date'].min().strftime('%Y-%m-%d'))
    
    if END_DATE:
        date_range.append(END_DATE.strftime('%Y-%m-%d'))
    else:
        date_range.append(df['Date'].max().strftime('%Y-%m-%d'))
    
    title += f" ({date_range[0]} to {date_range[1]})"

plt.title(title)
plt.xlabel('Date')
plt.tight_layout()

# Create a filename that includes the date range if specified
filename_suffix = ''
if START_DATE or END_DATE:
    start_str = START_DATE.strftime('%Y%m%d') if START_DATE else 'start'
    end_str = END_DATE.strftime('%Y%m%d') if END_DATE else 'end'
    filename_suffix = f"_{start_str}_to_{end_str}"

plt.savefig(f'plots/strategy/{TICKER}_sip_simulation{filename_suffix}.png')
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

with open(f'data/{TICKER}_sipNoExit_simulation{filename_suffix}_timeseries.json', 'w') as f:
    json.dump(output_data, f, indent=2)
