import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

script_name = os.path.splitext(os.path.basename(__file__))[0]

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
INDEX_FILE = os.path.join(DATA_DIR, 'SPX_sma_data.json')
SIP_FILE = os.path.join(DATA_DIR, 'SPX_sipNoExit_simulation_timeseries.json')
CROSSOVER_FILE = os.path.join(DATA_DIR, 'SPX_crossover_sip_simulation_timeseries.json')
REVERSE_CROSSOVER_FILE = os.path.join(DATA_DIR, 'SPX_reverse_crossover_sip_simulation_timeseries.json')

# Ensure the output directory exists
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots/strategy')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load index and SMA data
with open(INDEX_FILE, 'r') as f:
    index_data = json.load(f)
index_df = pd.DataFrame(index_data)
index_df['Date'] = pd.to_datetime(index_df['Date'])

strategy_files = [
    (SIP_FILE, 'SIP', 'green'),
    (CROSSOVER_FILE, 'Crossover', 'orange'),
    (REVERSE_CROSSOVER_FILE, 'Reverse Crossover', 'purple')
]

fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=True)

# 1. Index + SMA + Net Capital Invested
ax = axes[0]
ax.plot(index_df['Date'], index_df['Close'], label='SPX Index', color='black', linewidth=1.5)
ax.plot(index_df['Date'], index_df['SMA_50'], label='SMA 50', color='blue', linestyle='--')
ax.plot(index_df['Date'], index_df['SMA_200'], label='SMA 200', color='red', linestyle='--')
ax.set_ylabel('Index Value')
ax.set_title('SPX Index with SMA 50/200 and Net Capital Invested')
ax.grid(True, which='both', linestyle='--', alpha=0.5)

ax2 = ax.twinx()
for file, label, color in strategy_files:
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        dates = pd.to_datetime([d['Date'] for d in data])
        net_cap = [d['NetCapitalInvested'] for d in data]
        ax2.plot(dates, net_cap, label=f'Net Capital: {label}', color=color, linewidth=1.2, alpha=0.7)
ax2.set_ylabel('Net Capital Invested ($)')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 2. ROIC
ax_roic = axes[1]
for file, label, color in strategy_files:
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        dates = pd.to_datetime([d['Date'] for d in data])
        roic = [d['ROIC'] for d in data]
        ax_roic.plot(dates, roic, label=f'ROIC: {label}', color=color, linewidth=1.2, alpha=0.7)
ax_roic.set_ylabel('ROIC')
ax_roic.set_title('ROIC Over Time (All Strategies)')
ax_roic.grid(True, which='both', linestyle='--', alpha=0.5)
ax_roic.legend(loc='upper left')

# 3. Net Valuation
ax_val = axes[2]
for file, label, color in strategy_files:
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        dates = pd.to_datetime([d['Date'] for d in data])
        net_val = [d['NetValuation'] for d in data]
        ax_val.plot(dates, net_val, label=f'Net Valuation: {label}', color=color, linewidth=1.2, alpha=0.7)
ax_val.set_ylabel('Net Valuation ($)')
ax_val.set_xlabel('Date')
ax_val.set_title('Net Valuation Over Time (All Strategies)')
ax_val.grid(True, which='both', linestyle='--', alpha=0.5)
ax_val.legend(loc='upper left')

# Set x-axis major ticks to every year for all subplots
for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()

fig.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'{script_name}_3subplot_index_sma_netcap_roic_netval.png'))
plt.show()
