# This script implements a dynamic investment strategy with a trailing stop loss mechanism on a 2x leveraged index.
# It reads historical price and SMA data, simulates monthly investment decisions, and tracks portfolio performance.
# Key features:
# - Invests $1 if price > SMA1 > SMA2, or a fraction of cash if SMA1 < price < SMA2 (with increasing aggressiveness).
# - Uses a dynamic trailing stop loss based on ROIC thresholds to protect gains.
# - Only one investment action per month.
# - Tracks and plots net value, invested capital, cash, and ROIC over time.
# - Saves results to CSV and outputs a summary plot.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Initialize variables for portfolio and strategy state
cash = 1000  # Starting cash
holdings = 0  # Shares held
invested_capital = 0  # Total invested
net_value = cash  # Initial net value
highest_value = cash  # Track highest value for trailing stop
stop_loss = None  # Current stop loss percentage
stop_loss_value = None  # Stop loss value in dollars
trailing_stops = [0.34, 0.80, 1.40, 2.22, 3.23, 4.78, 6.75]  # Stop loss thresholds
factor = 0.1  # 10% of savings for conditional investment
n_count = 0  # Counter for SMA1 < price < SMA2 condition
monthly_action_done = False  # Flag to ensure one action per month
current_month = None  # Track current month
results = []  # Store metrics

# Reading the data from JSON file and preprocessing
file_path_2x = './data/SPX_2x.json'
data_2x = pd.read_json(file_path_2x)
data_2x['Date'] = pd.to_datetime(data_2x['Date'])
data_2x.set_index('Date', inplace=True)

# Filter data for the selected date range
start_date = pd.to_datetime('2000-01-01')
end_date = pd.to_datetime('2004-12-31')
data_2x = data_2x[(data_2x.index >= start_date) & (data_2x.index <= end_date)]
daily_data = data_2x.copy()

# Main investment simulation loop, iterating over each day
for date, row in daily_data.iterrows():
    price_2x = row['Close_xlev']
    sma1 = row['SMA_1']
    sma2 = row['SMA_2']
    
    # Check if we're in a new month and reset monthly action flag
    month = date.to_period('M')
    if month != current_month:
        monthly_action_done = False
        current_month = month
    
    # Skip if action already taken this month
    if monthly_action_done:
        continue
    
    # Calculate current portfolio value and ROIC
    portfolio_value = holdings * price_2x
    net_value = cash + portfolio_value
    roic = (net_value - invested_capital) / invested_capital if invested_capital > 0 else 0
    
    # Update trailing stop loss if holdings exist
    if holdings > 0:
        if net_value > highest_value:
            highest_value = net_value
            # Set new stop loss based on ROIC thresholds
            for stop in trailing_stops:
                if roic >= stop:
                    stop_loss = stop
                    stop_loss_value = invested_capital * (1 + stop)
                else:
                    break
        
        # Check if stop loss is triggered; liquidate if so
        if stop_loss is not None and net_value <= stop_loss_value:
            cash += portfolio_value  # Sell all holdings
            holdings = 0
            invested_capital = 0
            stop_loss = None
            stop_loss_value = None
            highest_value = cash
            monthly_action_done = True
    
    # Trading logic for new investments (only if no action this month)
    if not monthly_action_done:
        if price_2x > sma1 and sma1 > sma2:
            # Condition 1: Invest $1 when price > SMA1 > SMA2
            if cash >= 1:
                shares = 1 / price_2x
                holdings += shares
                cash -= 1
                invested_capital += 1
                monthly_action_done = True
        elif sma1 < price_2x < sma2:
            # Condition 3: Invest factor% of savings
            n_count += 1
            if n_count < 5:
                investment = cash * factor
                if cash >= investment:
                    shares = investment / price_2x
                    holdings += shares
                    cash -= investment
                    invested_capital += investment
                    monthly_action_done = True
            else:
                # After n=5, invest all cash
                if cash > 0:
                    shares = cash / price_2x
                    holdings += shares
                    invested_capital += cash
                    cash = 0
                    monthly_action_done = True
    
    # Store daily metrics for analysis and plotting
    results.append({
        'Date': date,
        'Net Value': net_value,
        'ROIC': roic,
        'Cash': cash,
        'Holdings': holdings,
        'Price': price_2x,
        'SMA1': sma1,
        'SMA2': sma2,
        'Stop Loss': stop_loss if stop_loss is not None else 0
    })

# Convert results to DataFrame for output and plotting
results_df = pd.DataFrame(results)
results_df.set_index('Date', inplace=True)

# Save results to CSV for further analysis
results_df.to_csv('trading_results.csv')

# Function to add vertical grid lines at the start of each month in plots
def add_month_start_grid(ax, dates):
    for date in dates[dates.is_month_start]:
        ax.axvline(x=date, color='gray', linestyle=':', linewidth=0.5)

# Plotting section: creates two subplots for performance and index/SMA
plt.figure(figsize=(14, 8))
gs = GridSpec(2, 1, height_ratios=[3, 1])

# Upper plot: Net Value, Invested Capital, Cash, and ROIC
ax1 = plt.subplot(gs[0])
ax1.plot(results_df.index, results_df['Net Value'], label='Net Value', color='blue')
ax1.plot(results_df.index, results_df['Net Value'] - results_df['Cash'], label='Invested Capital', color='orange')
ax1.plot(results_df.index, results_df['Cash'], label='Savings Account', color='green')
ax1.set_title('Net Value, Invested Capital, Savings Account, and ROIC Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add month start grid lines
add_month_start_grid(ax1, results_df.index)

ax1b = ax1.twinx()
ax1b.plot(results_df.index, results_df['ROIC'], label='ROIC', color='red', linestyle='--')
ax1b.set_ylabel('ROIC')
ax1b.set_ylim(min(results_df['ROIC']) * 1.1, max(results_df['ROIC']) * 1.1)
ax1b.legend(loc='upper right')

# Lower plot: 2x Index, SMA1, and SMA2
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(daily_data.index, daily_data['Close_xlev'], label='Lev Index', color='purple')
ax2.plot(daily_data.index, daily_data['SMA_1'], label='SMA1', color='red', linestyle=':')
ax2.plot(daily_data.index, daily_data['SMA_2'], label='SMA2', color='black', linestyle='-.')
ax2.set_title('2x Index, SMA1, and SMA2 Over Time')
ax2.set_xlabel('Date')
ax2.set_ylabel('2x Index / SMA')
ax2.legend()
ax2.grid(True)

# Add month start grid lines to the second plot
add_month_start_grid(ax2, daily_data.index)

plt.tight_layout()

# Create directory and save plot
os.makedirs('./plots/strategy', exist_ok=True)
plt.savefig('./plots/strategy/strategy_dynamic_TrailingSL.png')
plt.close()

# Print final metrics to console
print(f"Final Net Value: ${net_value:.2f}")
print(f"Final ROIC: {roic*100:.2f}%")
print(f"Final Cash: ${cash:.2f}")
print(f"Final Holdings: {holdings:.4f} shares")
print("Plot saved as './plots/strategy/strategy_dynamic_TrailingSL.png'")