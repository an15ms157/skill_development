import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.dates as mdates

TICKER = 'SPX'
DATA_PATH = f'data/{TICKER}_sma_data.json'
factor = 0.15  # Set catchup factor to 0.5

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"SMA data file for ticker {TICKER} not found at {DATA_PATH}")

with open(DATA_PATH, 'r') as f:
    records = json.load(f)

df = pd.DataFrame(records)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Only invest when SMA_50 > SMA_200
sma1 = 'SMA_50'
sma2 = 'SMA_200'
df['YearMonth'] = df['Date'].dt.to_period('M')
months = df['YearMonth'].unique()

np.random.seed(42)  # For reproducibility

invested_dates = []
invested_prices = []
invested_capital = []
units_held = []
net_valuation = []
roic = []
savings_over_time = []
total_invested = 0
units = 0
savings = 0.0
catchup_counter = 0
catchup_active = False

all_dates = []
all_invested_capital = []
all_net_valuation = []
all_roic = []
all_savings = []

net_catchup_months = int(1 / factor)  # Use factor variable

for month in months:
    month_df = df[df['YearMonth'] == month]
    idx = np.random.choice(month_df.index)
    invest_row = month_df.loc[idx]
    sma_50 = invest_row[sma1]
    sma_200 = invest_row[sma2]
    # Check if we can invest this month
    can_invest = pd.isna(sma_50) or pd.isna(sma_200) or (sma_50 > sma_200)
    if can_invest:
        # Always invest the usual $1
        invest_amount = 1.0
        catchup_part = 0.0
        if savings > 0 and (catchup_active or catchup_counter > 0):
            catchup_active = True
            catchup_counter += 1
            catchup_part = min(factor * savings, savings)
            invest_amount += catchup_part
            savings -= catchup_part
        elif savings > 0:
            # Start catch-up
            catchup_active = True
            catchup_counter = 1
            catchup_part = min(factor * savings, savings)
            invest_amount += catchup_part
            savings -= catchup_part
        else:
            catchup_active = False
            catchup_counter = 0
        # Prevent negative savings
        savings = max(savings, 0.0)
        # Prevent negative invest_amount
        invest_amount = max(invest_amount, 0.0)
        # Invest
        price = invest_row['Close']
        date = invest_row['Date']
        units += invest_amount / price
        total_invested += invest_amount
        # Ensure total_invested never decreases
        total_invested = max(total_invested, 0.0)
        invested_dates.append(date)
        invested_prices.append(price)
        invested_capital.append(total_invested)
        units_held.append(units)
        # Net valuation at this point (using investment date's close)
        value = units * price
        net_valuation.append(value)
        roic.append((value - total_invested) / total_invested if total_invested > 0 else 0)
        # Stop catch-up after net_catchup_months or if savings depleted
        if catchup_active and (catchup_counter >= net_catchup_months or savings < 1e-8):
            # Use up all remaining savings in the last catch-up month
            if savings > 0:
                invest_amount += savings
                total_invested += savings
                units += savings / price
                savings = 0.0
            catchup_active = False
            catchup_counter = 0
    else:
        # Save the $1 for this month
        savings += 1.0
        savings = max(savings, 0.0)
        catchup_active = False
        catchup_counter = 0
    # Track for all months
    date = invest_row['Date']
    all_dates.append(date)
    all_invested_capital.append(total_invested)
    # Use investment date's price if invested, else use month's last close
    if can_invest:
        value = units * price
    else:
        month_last_price = month_df['Close'].iloc[-1]
        value = units * month_last_price
    all_net_valuation.append(value)
    all_roic.append((value - total_invested) / total_invested if total_invested > 0 else 0)
    all_savings.append(savings)
    # Assertions for debugging
    assert total_invested >= 0, f"total_invested negative: {total_invested}"
    assert savings >= 0, f"savings negative: {savings}"
    assert all_invested_capital[-1] >= 0, f"NetCapitalInvested negative: {all_invested_capital[-1]}"

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
ax1.plot(all_dates, all_invested_capital, label='Net Capital Invested', color='blue')
ax1.plot(all_dates, all_net_valuation, label='Net Valuation', color='green')
ax1.plot(all_dates, all_savings, label='Savings', color='orange', linestyle='--')
ax1.set_ylabel('Dollars')
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

# After plotting but before legend and title
ax1.set_ylim(bottom=-10)
ax2.set_ylim(bottom=-.1)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_2 + lines_1, labels_2 + labels_1, loc='upper left')

plt.title(f'{TICKER} Crossover SIP: Capital Invested, Net Valuation, Savings, and ROIC Over Time')
ax2.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
fig.tight_layout()
plt.savefig(f'plots/strategy/{TICKER}_crossover_sip_simulation.png')
plt.show()

#log
print("Date".ljust(12), "NetCapitalInvested".rjust(18), "NetValuation".rjust(15), "ROIC".rjust(10), "Savings".rjust(10))
for date, capital, valuation, r, s in zip(all_dates, all_invested_capital, all_net_valuation, all_roic, all_savings):
    print(
        date.strftime('%Y-%m-%d').ljust(12),
        f"{capital:18.2f}",
        f"{valuation:15.2f}",
        f"{r:10.4f}",
        f"{s:10.2f}"
    )

# Save data to JSON based on time
output_data = []
for date, capital, valuation, r, s in zip(all_dates, all_invested_capital, all_net_valuation, all_roic, all_savings):
    output_data.append({
        'Date': date.strftime('%Y-%m-%d'),
        'NetCapitalInvested': capital,
        'NetValuation': valuation,
        'ROIC': r,
        'Savings': s
    })

with open(f'data/{TICKER}_crossover_sip_simulation_timeseries.json', 'w') as f:
    json.dump(output_data, f, indent=2)
