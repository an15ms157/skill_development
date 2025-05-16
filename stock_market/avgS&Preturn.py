import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd

# Define the stock ticker and the period of data retrieval
symbol = "^GSPC"  # S&P 500 index symbol on Yahoo Finance
start_date = "1940-01-01"  # Start date for historical data
end_date = "1950-01-01"  # Set the desired end date for historical data (this can be adjusted)

# Fetch the historical data using yfinance
data = yf.download(symbol, start=start_date, end=end_date)

# Ensure data contains the necessary columns
if not {"Open", "Close"}.issubset(data.columns):
    raise ValueError("The fetched data does not contain 'Open' and 'Close' columns.")

# Handle cases where Open price is 0 or dates before 1963
data['Adjusted_Open'] = data['Open'].copy()
data['Adjusted_Open'] = data['Adjusted_Open'].mask(data['Adjusted_Open'] == 0, data['Close'].shift(1))
data['Adjusted_Open'] = data['Adjusted_Open'].mask(data.index < '1963-01-01', data['Close'].shift(1))

# Check if start_date is available in the fetched data
if start_date not in data.index.strftime('%Y-%m-%d'):
    # If not, take the first available date in the data
    start_date = data.index[0].strftime('%Y-%m-%d')

# Function to create an index based on a multiplier
def create_index(multiplier=2):
    new_index = [data.loc[start_date, 'Close']]  # Start the index at the first S&P 500 close price
    dates = data.index
    for i, date in enumerate(dates[1:]):  # Skip the first day (already used as starting value)
        # Calculate the percentage change
        pct_change = (data.loc[date, 'Close'] - data.loc[dates[i], 'Close']) / data.loc[dates[i], 'Close']
        # Update the new index value with the multiplier
        new_index.append(new_index[-1] * (1 + multiplier * pct_change))
    return pd.Series(new_index, index=dates)

# Create the 2x and 3x indices
new_index_2x_series = create_index(multiplier=2)
new_index_3x_series = create_index(multiplier=3)

# Calculate the 200-day moving average for the 2x and 3x indices
moving_avg_200_2x = new_index_2x_series.rolling(window=200).mean()
moving_avg_200_3x = new_index_3x_series.rolling(window=200).mean()

# Plot the 3x index with the 200-day moving average
plt.figure(figsize=(12, 6))
plt.plot(new_index_3x_series, label='3x Daily Percentage Change Index', color='green', alpha=0.7)
plt.plot(moving_avg_200_3x, label='200-Day Moving Average', color='red', linestyle='--', alpha=0.7)
plt.title('3x Daily Percentage Change Index with 200-Day Moving Average', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Index Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# Save the 3x index plot
plt.savefig('3x_index_with_200_day_moving_avg.png', dpi=300)
plt.close()

# Plot the 3x index with the 200-day moving average
plt.figure(figsize=(12, 6))
plt.plot(new_index_2x_series, label='2x Daily Percentage Change Index', color='green', alpha=0.7)
plt.plot(moving_avg_200_2x, label='200-Day Moving Average', color='red', linestyle='--', alpha=0.7)
plt.title('2x Daily Percentage Change Index with 200-Day Moving Average', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Index Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# Save the 3x index plot
plt.savefig('2x_index_with_200_day_moving_avg.png', dpi=300)
plt.close()

# Plot all three indices: S&P 500, 2x index, and 3x index
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='S&P 500 Index (SPX)', color='cyan', alpha=0.5)
plt.plot(new_index_2x_series, label='2x Daily Percentage Change Index', color='blue', alpha=0.7)
plt.plot(new_index_3x_series, label='3x Daily Percentage Change Index', color='green', alpha=0.7)
plt.title('S&P 500 Index vs 2x & 3x Daily Percentage Change Indices', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Index Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# Save the plot with all three indices
plt.savefig('all_indices_plot.png', dpi=300)
plt.close()

