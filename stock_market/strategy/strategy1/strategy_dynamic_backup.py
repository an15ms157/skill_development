import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Reading the data
file_path_2x = './data/SPX_2x.json'
data_2x = pd.read_json(file_path_2x)
data_2x['Date'] = pd.to_datetime(data_2x['Date'])
data_2x.set_index('Date', inplace=True)

# Filter data_2x for the selected date range
start_date = pd.to_datetime('2003-01-01')
end_date = pd.to_datetime('2004-12-31')
data_2x = data_2x[(data_2x.index >= start_date) & (data_2x.index <= end_date)]



# Initializing variables
savings_account = 0
invested_capital = 0
shares_2x = 0
results = []
factor = 0.3  # Define your factor here
min_sav = 5    # Minimum savings threshold
last_invested_month = None
first_below_sma2 = False
can_invest_2x = True

# Investment loop
for date, row in data_2x.iterrows():
    price_2x = row['Close_2x']
    sma1 = row['SMA_2x_300']
    sma2 = row['SMA_2x_600']

    if pd.isna(sma2) or price_2x is None:
        continue

    # Invest or deposit only once per month, on the first encountered trading day of the month
    current_month = (date.year, date.month)
    invest_this_month = False
    investment_amount_this_month = 0
    if last_invested_month != current_month:
        invest_this_month = True
        last_invested_month = current_month
        invested_capital += 1
        if can_invest_2x:
            if price_2x >= sma2:  # Only invest if price_2x >= SMA_2x_600
                # Invest $0.7 in the 2x index and save $0.3
                shares_2x += 0.7 / price_2x
                savings_account += 0.3
                investment_amount_this_month = 0.7
            else:
                # Save $1 in the savings account
                savings_account += 1
                investment_amount_this_month = 0
        else:
            # Save $1 in the savings account if not allowed to invest in 2x
            savings_account += 1
            investment_amount_this_month = 0

    # Check if 2x price is less than SMA2
    if can_invest_2x and price_2x < sma2:
        # Case 1: If SMA1 > SMA2, sell all 2x holdings
        if sma1 > sma2:
            savings_account += shares_2x * price_2x
            shares_2x = 0
            first_below_sma2 = True
            can_invest_2x = False  # Block further 2x investments until criteria resurfaces
            investment_amount_this_day = 0
            index_2x_value = 0
            net_value = savings_account
            roic = (net_value - invested_capital) / invested_capital if invested_capital > 0 else 0
            results.append({
                'Date': date,
                'Invested_Capital': invested_capital,
                'Net_Value': net_value,
                'Savings_Account': savings_account,
                'ROIC': roic,
                '2x_Holding': 0,
                'Investment_Amount': 0,
                'Price_2x': price_2x,
                'SMA1': sma1,
                'SMA2': sma2
            })
            continue
        # Case 2: If price < SMA2 but SMA1 < SMA2, don't sell 2x holdings
        # Just mark first_below_sma2 as True to enable factor investment when price rebounds
        else:
            first_below_sma2 = True

    # Allow 2x investment again ONLY if price_2x >= sma1 and a new monthly investment is triggered
    if not can_invest_2x and price_2x >= sma1 and invest_this_month:
        can_invest_2x = True

    investment_amount_this_day = 0
    if can_invest_2x and first_below_sma2 and price_2x >= sma1:  # Only invest from savings if price_2x >= SMA_2x_300
        investment_amount = factor * savings_account
        if investment_amount > 0:
            shares_2x += investment_amount / price_2x
            savings_account -= investment_amount
            investment_amount_this_day = investment_amount

    # Check minimum savings threshold
    if savings_account < min_sav:
        investment_amount = savings_account
        shares_2x += investment_amount / price_2x
        savings_account = 0
        investment_amount_this_day = investment_amount

    # Calculate net value
    if not can_invest_2x:
        index_2x_value = 0
    else:
        index_2x_value = shares_2x * price_2x
    net_value = index_2x_value + savings_account

    # Calculate ROIC
    roic = (net_value - invested_capital) / invested_capital if invested_capital > 0 else 0

    # Store results
    results.append({
        'Date': date,
        'Invested_Capital': invested_capital,
        'Net_Value': net_value,
        'Savings_Account': savings_account,
        'ROIC': roic,
        '2x_Holding': index_2x_value,
        'Investment_Amount': investment_amount_this_day,
        'Price_2x': price_2x,
        'SMA1': sma1,
        'SMA2': sma2
    })

# Creating DataFrame
results_df = pd.DataFrame(results)
results_df.set_index('Date', inplace=True)

# Set display options for maximum width and rows
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
print("Results with Price_2x, SMA1, and SMA2 data for 2000-01-01 to 2001-12-31:")
print(results_df.loc[start_date:end_date])

# Plotting
plt.figure(figsize=(14, 8))

gs = GridSpec(2, 1, height_ratios=[3, 1])

# Upper plot: Net Value, Invested Capital, Savings Account (left y-axis) and ROIC (right y-axis)
ax1 = plt.subplot(gs[0])
ax1.plot(results_df.index, results_df['Net_Value'], label='Net Value', color='blue')
ax1.plot(results_df.index, results_df['Invested_Capital'], label='Invested Capital', color='orange')
ax1.plot(results_df.index, results_df['Savings_Account'], label='Savings Account', color='green')
ax1.set_title('Net Value, Invested Capital, Savings Account, and ROIC Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)

ax1b = ax1.twinx()
ax1b.plot(results_df.index, results_df['ROIC'], label='ROIC', color='red', linestyle='--')
ax1b.set_ylabel('ROIC')
ax1b.set_ylim(0, 10)
ax1b.legend(loc='upper right')

# Lower plot: 2x Index, SMA1, and SMA2
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(data_2x.index, data_2x['Close_2x'], label='2x Index', color='purple')
ax2.plot(data_2x.index, data_2x['SMA_2x_300'], label='SMA1 (300)', color='brown', linestyle=':')
ax2.plot(data_2x.index, data_2x['SMA_2x_600'], label='SMA2 (600)', color='black', linestyle='-.')
ax2.set_title('2x Index, SMA1, and SMA2 Over Time')
ax2.set_xlabel('Date')
ax2.set_ylabel('2x Index / SMA')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# Creating directory if it doesn't exist
os.makedirs('./plots/strategy', exist_ok=True)

# Saving the plot
plt.savefig('./plots/strategy/strategy_dynamic_combined.png')
plt.close()

print("Plot saved as './plots/strategy/strategy_dynamic.png'")
