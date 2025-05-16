# strategy_crossover_2x_only_when_sma50_gt_sma200.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.dates as mdates

TICKER = 'SPX'
DATA_PATH = f'data/{TICKER}_sma_data.json'
LEV_DATA_PATH = f'data/{TICKER}_2x.json'  # 2x leveraged index data
factor = 0.15  # Factor for main index investment

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"SMA data file for ticker {TICKER} not found at {DATA_PATH}")
if not os.path.exists(LEV_DATA_PATH):
    raise FileNotFoundError(f"2x data file for ticker {TICKER} not found at {LEV_DATA_PATH}")

with open(DATA_PATH, 'r') as f:
    records = json.load(f)
with open(LEV_DATA_PATH, 'r') as f:
    lev_records = json.load(f)

df = pd.DataFrame(records)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
lev_df = pd.DataFrame(lev_records)
lev_df['Date'] = pd.to_datetime(lev_df['Date'])
lev_df = lev_df.sort_values('Date').reset_index(drop=True)
lev_df = lev_df.set_index('Date')

sma1 = 'SMA_50'
sma2 = 'SMA_200'
df['YearMonth'] = df['Date'].dt.to_period('M')
months = df['YearMonth'].unique()

np.random.seed(42)

main_units = 0.0
lev_units = 0.0
main_invested = 0.0
lev_invested = 0.0

all_dates = []
all_main_invested = []
all_lev_invested = []
all_main_valuation = []
all_lev_valuation = []
all_total_valuation = []
all_roic = []
all_factor_transferred = []

catchup_counter = 0
catchup_months = int(1 / factor)
catchup_active = False
prev_crossover = False  # Track previous crossover state

for month in months:
    month_df = df[df['YearMonth'] == month]
    idx = np.random.choice(month_df.index)
    invest_row = month_df.loc[idx]
    date = invest_row['Date']
    sma_50 = invest_row[sma1]
    sma_200 = invest_row[sma2]
    main_price = invest_row['Close']
    # Get 2x price for the same date (or closest available)
    if date in lev_df.index:
        lev_price = lev_df.loc[date]['Close_2x']
    else:
        nearest_idx = lev_df.index.get_indexer([date], method='nearest')[0]
        lev_price = lev_df.iloc[nearest_idx]['Close_2x']
    # Only invest in main index when SMA_50 > SMA_200
    factor_transferred = 0.0
    crossover = pd.notna(sma_50) and pd.notna(sma_200) and (sma_50 > sma_200)
    if crossover:
        # Start catchup only on crossover event
        if not prev_crossover:
            catchup_active = True
            catchup_counter = 0
        # Invest $1 in main index
        main_units += 1.0 / main_price
        main_invested += 1.0
        if catchup_active and catchup_counter < catchup_months:
            lev_value = lev_units * lev_price
            factor_transferred = factor * lev_value
            if lev_units > 0 and factor_transferred > 0:
                units_to_sell = factor_transferred / lev_price
                lev_units -= units_to_sell
                lev_value = lev_units * lev_price
                main_units += factor_transferred / main_price
                main_invested += factor_transferred
            catchup_counter += 1
            if catchup_counter >= catchup_months:
                catchup_active = False
        else:
            factor_transferred = 0.0
    else:
        # Otherwise, invest $1 in 2x index
        lev_units += 1.0 / lev_price
        lev_invested += 1.0
        catchup_active = False
        catchup_counter = 0
        factor_transferred = 0.0
    prev_crossover = crossover  # Update previous crossover state
    # Track values
    main_value = main_units * main_price
    lev_value = lev_units * lev_price
    total_value = main_value + lev_value
    total_invested = main_invested + lev_invested
    roic = (total_value - total_invested) / total_invested if total_invested > 0 else 0
    all_dates.append(date)
    all_main_invested.append(main_invested)
    all_lev_invested.append(lev_invested)
    all_main_valuation.append(main_value)
    all_lev_valuation.append(lev_value)
    all_total_valuation.append(total_value)
    all_roic.append(roic)
    all_factor_transferred.append(factor_transferred)

# Plotting
plt.style.use('default')
fig, ax2 = plt.subplots(figsize=(12, 6))

ax2.plot(all_dates, all_roic, label='ROIC', color='purple')
ax2.set_ylabel('ROIC')
ax2.set_xlabel('Date')
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()

ax1 = ax2.twinx()
ax1.plot(all_dates, [main+lev for main,lev in zip(all_main_invested, all_lev_invested)], label='Index Invested', color='blue')
ax1.plot(all_dates, all_main_valuation, label='Main Index Value', color='green')
ax1.plot(all_dates, all_lev_valuation, label='2x Index Value', color='red')
ax1.plot(all_dates, all_total_valuation, label='Total Value', color='black', linestyle='--')
ax1.set_ylabel('Dollars')
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

ax1.set_ylim(bottom=-10)
ax2.set_ylim(bottom=-.1)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_2 + lines_1, labels_2 + labels_1, loc='upper left')

plt.title(f'{TICKER} Crossover 2x: Only Invest in Main Index When SMA50 > SMA200')
ax2.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
fig.tight_layout()
plt.savefig(f'plots/strategy/{TICKER}_crossover_2x.png')
plt.show()

# Save data to JSON
output_data = []
for date, main_inv, lev_inv, main_val, lev_val, total_val, r, ftrans in zip(
        all_dates, all_main_invested, all_lev_invested, all_main_valuation, all_lev_valuation, all_total_valuation, all_roic, all_factor_transferred):
    index_invested = main_inv + lev_inv
    output_data.append({
        'Date': date.strftime('%Y-%m-%d'),
        'IndexInvested': index_invested,
        'MainIndexValue': main_val,
        '2xIndexValue': lev_val,
        'FactorTransferred': ftrans,
        'TotalValue': total_val,
        'ROIC': r
    })
with open(f'data/{TICKER}_crossover_2x_only_when_sma50_gt_sma200_timeseries.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("Date".ljust(12), "IndexInvested".rjust(15), "MainIndexValue".rjust(16), "2xIndexValue".rjust(14), "FactorTransferred".rjust(18), "TotalValue".rjust(14), "ROIC".rjust(10))
for date, main_inv, lev_inv, main_val, lev_val, total_val, r, ftrans in zip(
        all_dates, all_main_invested, all_lev_invested, all_main_valuation, all_lev_valuation, all_total_valuation, all_roic, all_factor_transferred):
    index_invested = main_inv + lev_inv
    print(
        date.strftime('%Y-%m-%d').ljust(12),
        f"{index_invested:15.2f}",
        f"{main_val:16.2f}",
        f"{lev_val:14.2f}",
        f"{ftrans:18.2f}",
        f"{total_val:14.2f}",
        f"{r:10.4f}"
    )