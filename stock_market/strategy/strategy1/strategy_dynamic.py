import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# Load 2x index data
with open('./data/SPX_2x.json', 'r') as f:
    records = json.load(f)
df = pd.DataFrame(records)
df['Date'] = pd.to_datetime(df['Date'])

# Parameters
sma_col = 'SMA_2x_600'  # SMA2 (assumed to be 2-day SMA, but check your data)
monthly_invest = 1.0

# State variables
lev2x_units = 0.0
saved_cash = 0.0
total_invested = 0.0

# For results
history = []

# Group by month for monthly investment
monthly_groups = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

for (year, month), group in monthly_groups:
    # Pick a random day in this month
    day = group.sample(1, random_state=year*100+month).iloc[0]
    date = day['Date']
    price = day['Close_2x']  # 2x index price
    sma2 = day[sma_col]
    sma1 = day['SMA_2x_300']  # Assuming SMA1 is 300-period SMA, adjust if necessary
    
    # Debugging output
    print(f"Date: {date}, Price: {price:.2f}, SMA2: {sma2:.2f}, Condition (price > sma2): {price > sma2}")
    # Investment logic
    # Treat NaN SMA as a condition to invest $1
    if pd.isna(sma2) or price > sma2:
        # Invest $1 in 2x index
        lev2x_units += monthly_invest / price
        total_invested += monthly_invest
        action = f'Invest $1 in 2x at {price:.2f}'
    elif sma2 >= price:
        # Pull out cash from 2x index
        if lev2x_units > 0:
            cash_out = lev2x_units * price
            saved_cash += cash_out
            lev2x_units = 0  # Reset units after pulling out cash
            action = f'Pulled out ${cash_out:.2f} from 2x index, total saved: ${saved_cash:.2f}'
        else:
            action = 'No units in 2x index to pull out cash.'
    elif sma2 > 2 * price > sma1:
        # Restart investing in main index and 2x index
        n_months = 10  # Number of months to deplete savings
        main_investment = saved_cash / (2 * n_months)  # Divide savings into 2n parts
        lev2x_investment = saved_cash / (2 * n_months) + monthly_invest  # Add $1 to each n part for 2x index

        # Invest in main index
        main_index_units = main_investment / price
        saved_cash -= main_investment

        # Invest in 2x index
        lev2x_units += lev2x_investment / price
        total_invested += lev2x_investment
        saved_cash -= lev2x_investment

        action = (f'Restarted investing: ${main_investment:.2f} in main index and '
                  f'${lev2x_investment:.2f} in 2x index at {price:.2f}, '
                  f'remaining savings: ${saved_cash:.2f}')
    else:
        # Save $1 (do not add to total_invested)
        saved_cash += monthly_invest
        action = f'Save $1, total saved: ${saved_cash:.2f}'
    
    # Calculate net value
    net_value = (lev2x_units * price) + saved_cash
    roic = (net_value - total_invested) / total_invested if total_invested > 0 else 0
    
    # Record history
    history.append({
        'Date': date,
        'Invested_Capital': total_invested,
        'Net_Value': net_value,
        'ROIC': roic,
        'Saved_Cash': saved_cash,
        'Action': action
    })

# Create DataFrame for results
monthly = pd.DataFrame(history)
monthly['Date'] = pd.to_datetime(monthly['Date'])

# Print all values
print(monthly[['Date', 'Invested_Capital', 'Net_Value', 'ROIC', 'Saved_Cash']].to_string(index=False))

# Add columns for 2x index value and main index value
monthly['2x_Index_Value'] = monthly['Net_Value'] - monthly['Saved_Cash']
monthly['Main_Index_Value'] = monthly['Saved_Cash']

# Print the values of 2x index and main index over the months
print(monthly[['Date', '2x_Index_Value', 'Main_Index_Value']].to_string(index=False))

# Modify the existing plot to include 2x index value and main index value
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Top: Invested Capital, Net Value, Saved Cash, 2x Index Value, Main Index Value
axs[0].plot(monthly['Date'], monthly['Invested_Capital'], label='Invested Capital', color='blue')
axs[0].plot(monthly['Date'], monthly['Net_Value'], label='Net Value', color='black', linestyle='--')
axs[0].plot(monthly['Date'], monthly['Saved_Cash'], label='Saved Cash', color='green')
axs[0].plot(monthly['Date'], monthly['2x_Index_Value'], label='2x Index Value', color='red')
axs[0].plot(monthly['Date'], monthly['Main_Index_Value'], label='Main Index Value', color='orange')
axs[0].set_yscale('log')  # Set y-axis to log scale
axs[0].set_ylabel('Value ($)')
axs[0].set_title('Invested Capital, Net Value, Saved Cash, 2x Index Value, and Main Index Value')
axs[0].legend()
axs[0].grid(True)

# Bottom: ROIC
axs[1].plot(monthly['Date'], monthly['ROIC'], label='ROIC', color='purple', linestyle=':')
axs[1].set_ylabel('ROIC')
axs[1].set_xlabel('Date')
axs[1].set_title('Return on Invested Capital (ROIC)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('./plots/strategy/strategy_dynamic_combined.png')
plt.show()