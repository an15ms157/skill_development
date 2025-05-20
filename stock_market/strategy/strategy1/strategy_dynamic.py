import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

"""
Dynamic Stock Market Investment Strategy Simulation

This script simulates a dynamic investment strategy in a 2x leveraged index with the following rules:
1. Invests $1 per month based on price and moving average conditions
2. Sells all 2x holdings if SMA1 > SMA2 > price and blocks further investments until price > SMA1
3. Invests monthly when price > SMA2
4. Uses savings strategically when SMA1 < price < SMA2
5. Maintains a minimum savings threshold of invested capital
6. Invests 30% of excess savings when above threshold, all remaining savings when below threshold
7. Tracks net value, return on invested capital (ROIC), and other metrics
8. Executes only one action per month, then stops until the next month

The strategy aims to optimize returns by timing investments based on price relative to moving averages.
"""

# Function to add vertical grid lines at month starts
def add_month_start_grid(ax, date_index, color='gray', linestyle='-', alpha=0.3):
    # Get the first day of each month in the date range
    month_starts = [date for date in date_index if date.day == 1]
    
    # Add vertical lines for month starts
    for date in month_starts:
        ax.axvline(x=date, color=color, linestyle=linestyle, alpha=alpha)
    
    # Format x-axis to show month and year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    # Rotate date labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Reading the data
file_path_2x = './data/SPX_3x.json'
data_2x = pd.read_json(file_path_2x)
data_2x['Date'] = pd.to_datetime(data_2x['Date'])
data_2x.set_index('Date', inplace=True)

# Filter data_2x for the selected date range
start_date = pd.to_datetime('1950-01-01')
end_date = pd.to_datetime('1980-12-31')
data_2x = data_2x[(data_2x.index >= start_date) & (data_2x.index <= end_date)]

# Using daily data directly instead of resampling to monthly
daily_data = data_2x.copy()

# Initializing variables
savings_account = 0  # Start with zero savings
invested_capital = 0
shares_2x = 0
results = []
factor = 0.4  # Factor for investing from savings above threshold
can_invest_2x = True
monthly_investment = 1  # $1 per month
savings_reinvest_counter = 0  # Counter for tracking savings re-investments
savings_reinvest_counter_max = 5  # Maximum allowed re-investments before full investment

# Track current month to implement one action per month rule
current_month = None
action_taken_this_month = False

# Investment loop - using daily data
for date, row in daily_data.iterrows():
    price_2x = row['Close_xlev']
    sma1 = row['SMA_1']
    sma2 = row['SMA_2']

    if pd.isna(sma2) or price_2x is None:
        continue
    
    # Check if we're in a new month
    if current_month != date.month:
        current_month = date.month
        action_taken_this_month = False
        # Add monthly investment at the start of each month
        invested_capital += monthly_investment
    
    # Initialize daily tracking variables
    investment_amount_this_day = 0
    investment_amount_this_day_from_savings = 0
    
    # Only execute trading logic if we haven't taken an action this month
    if not action_taken_this_month:
        # Rule 1: Sell 2x and save all money if SMA1 > SMA2 > price
        if sma1 > sma2 and sma2 > price_2x and shares_2x > 0:
            savings_account += shares_2x * price_2x
            shares_2x = 0
            can_invest_2x = False  # Block further 2x investments
            action_taken_this_month = True
        
        # Check if we can invest in 2x again
        elif not can_invest_2x and price_2x > sma1:
            can_invest_2x = True
            action_taken_this_month = True
        
        # Decision for the monthly investment
        elif can_invest_2x and price_2x > sma2:
            # Rule 3: Invest monthly amount when price > SMA2
            shares_2x += monthly_investment / price_2x
            investment_amount_this_day = monthly_investment
            
            # Investment logic for savings based on threshold
            if savings_account > 0:
                # Check if we've reached the max reinvestment count
                if savings_reinvest_counter >= savings_reinvest_counter_max:
                    # Invest all remaining savings
                    investment_amount = savings_account
                    shares_2x += investment_amount / price_2x
                    investment_amount_this_day_from_savings = investment_amount
                    savings_account = 0
                    savings_reinvest_counter = 0  # Reset counter
                # If savings above threshold, invest a fixed factor of excess
                else:
                    investment_amount = factor * (savings_account)
                    shares_2x += investment_amount / price_2x
                    investment_amount_this_day_from_savings = investment_amount
                    savings_account -= investment_amount
                    savings_reinvest_counter += 1  # Increment counter
            
            action_taken_this_month = True
        
        # Rule 2: Invest from savings when SMA1 < price < SMA2
        elif can_invest_2x and sma1 < price_2x and price_2x < sma2:
            # Save the monthly investment
            savings_account += monthly_investment
            
            # Investment logic for savings based on threshold
            if savings_account > 0:
                # Check if we've reached the max reinvestment count
                if savings_reinvest_counter >= savings_reinvest_counter_max:
                    # Invest all remaining savings
                    investment_amount = savings_account
                    shares_2x += investment_amount / price_2x
                    investment_amount_this_day_from_savings = investment_amount
                    savings_account = 0
                    savings_reinvest_counter = 0  # Reset counter
                # If savings above threshold, invest a fixed factor of excess
                else:
                    investment_amount = factor * (savings_account)
                    shares_2x += investment_amount / price_2x
                    investment_amount_this_day_from_savings = investment_amount
                    savings_account -= investment_amount
                    savings_reinvest_counter += 1  # Increment counter
            
            action_taken_this_month = True
        
        # Default action: Save monthly investment if no other conditions met
        elif not action_taken_this_month:
            savings_account += monthly_investment
            action_taken_this_month = True
    
    # Calculate net value
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
        'Investment_From_Savings': investment_amount_this_day_from_savings,
        'Price_2x': price_2x,
        'SMA1': sma1,
        'SMA2': sma2,
        'Action_Taken': action_taken_this_month
    })
    
# Creating DataFrame
results_df = pd.DataFrame(results)
results_df.set_index('Date', inplace=True)

# Set display options for maximum width and rows
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
print(f"Daily Results Summary with Price_2x, SMA1, and SMA2 data for {start_date.date()} to {end_date.date()}:")
# Print all but the last column
# Filter results_df to include only the first entry of each month
# Print the monthly data
monthly_results_df = results_df[~results_df.index.to_period('M').duplicated(keep='first')]
print(monthly_results_df.iloc[:, :-1])

# Print first and last values of the metrics
print("\n===== FIRST AND LAST VALUES OF METRICS =====")
print("\nFIRST VALUES:")
first_row = results_df.iloc[0]
print(f"Date: {results_df.index[0].strftime('%Y-%m-%d')}")
print(f"Invested Capital: ${first_row['Invested_Capital']:.2f}")
print(f"Net Value: ${first_row['Net_Value']:.2f}")
print(f"Savings Account: ${first_row['Savings_Account']:.2f}")
print(f"ROIC: {first_row['ROIC']:.4f}")
print(f"2x Holding: ${first_row['2x_Holding']:.2f}")
print(f"Price_2x: ${first_row['Price_2x']:.2f}")
print(f"SMA1: ${first_row['SMA1']:.2f}")
print(f"SMA2: ${first_row['SMA2']:.2f}")

print("\nLAST VALUES:")
last_row = results_df.iloc[-1]
print(f"Date: {results_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Invested Capital: ${last_row['Invested_Capital']:.2f}")
print(f"Net Value: ${last_row['Net_Value']:.2f}")
print(f"Savings Account: ${last_row['Savings_Account']:.2f}")
print(f"ROIC: {last_row['ROIC']:.4f}")
print(f"2x Holding: ${last_row['2x_Holding']:.2f}")
print(f"Price_2x: ${last_row['Price_2x']:.2f}")
print(f"SMA1: ${last_row['SMA1']:.2f}")
print(f"SMA2: ${last_row['SMA2']:.2f}")

# Plotting
plt.figure(figsize=(14, 8))

gs = GridSpec(2, 1, height_ratios=[3, 1])

# Upper plot: Net Value, Invested Capital, Savings Account (left y-axis) and ROIC (right y-axis)
ax1 = plt.subplot(gs[0])
ax1.plot(results_df.index, results_df['Net_Value'], label='Net Value', color='blue')
ax1.plot(results_df.index, results_df['Invested_Capital'], label='Invested Capital', color='orange')
ax1.plot(results_df.index, results_df['Savings_Account'], label='Savings Account', color='green')
ax1.set_title('Net Value, Invested Capital, Savings Account, and ROIC Over Time (Daily)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Value (log scale)')
ax1.legend(loc='upper left')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid for both major and minor ticks

# Add month start grid lines
add_month_start_grid(ax1, results_df.index)

ax1b = ax1.twinx()
ax1b.plot(results_df.index, results_df['ROIC'], label='ROIC', color='red', linestyle='--')
ax1b.set_ylabel('ROIC')
ax1b.set_ylim(0, max(results_df['ROIC']) * 1.1)  # Dynamic scale based on max ROIC
ax1b.legend(loc='upper right')

# Lower plot: 2x Index, SMA1, and SMA2
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(daily_data.index, daily_data['Close_xlev'], label='Lev Index', color='purple')
ax2.plot(daily_data.index, daily_data['SMA_1'], label='SMA1 ', color='brown', linestyle=':')
ax2.plot(daily_data.index, daily_data['SMA_2'], label='SMA2 ', color='black', linestyle='-.')
ax2.set_title('2x Index, SMA1, and SMA2 Over Time (Daily)')
ax2.set_xlabel('Date')
ax2.set_ylabel('2x Index / SMA')
ax2.legend()
ax2.grid(True)

# Add month start grid lines to the second plot as well
add_month_start_grid(ax2, daily_data.index)

plt.tight_layout()

# Creating directory if it doesn't exist
os.makedirs('./plots/strategy', exist_ok=True)

# Saving the plot
plt.savefig('./plots/strategy/strategy_dynamic_monthly.png')
plt.close()

print("Plot saved as './plots/strategy/strategy_dynamic_monthly.png'")
