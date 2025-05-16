import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import strategy  # Importing the strategy module

# Define parameters
ticker = "^GSPC"
start_date = "2010-01-01"
end_date = "2024-04-11"
N = 500  # n for the moving average

# Download data
data = yf.download(ticker, start=start_date, end=end_date, progress=False)
price = data['Close']

print(f"Data range: {price.index[0]} to {price.index[-1]}")

# Calculate daily price change
price_change = price.diff()

# Initialize 2x and 3x synthetic indices with the same starting value
index_2x = pd.Series(index=price.index, dtype='float64')
index_3x = pd.Series(index=price.index, dtype='float64')

index_2x.iloc[0] = price.iloc[0]
index_3x.iloc[0] = price.iloc[0]

# Calculate synthetic leveraged indices using point change logic
for i in range(1, len(price)):
    delta = price_change.iloc[i]
    index_2x.iloc[i] = index_2x.iloc[i-1] + 2 * delta
    index_3x.iloc[i] = index_3x.iloc[i-1] + 3 * delta

# Calculate moving averages
ma_index = price.rolling(window=N).mean()
ma_2x = index_2x.rolling(window=N).mean()
ma_3x = index_3x.rolling(window=N).mean()

# Calculate n/2 moving average
ma_2x_half = index_2x.rolling(window=(N//2)).mean()

# Save CSVs for original, 2x, and 3x indices
original_data = pd.concat([price, ma_index], axis=1)
original_data.columns = ['Price', 'MA_Index']
original_data.dropna().to_csv('original_index.csv')
print(f"original_index.csv saved with shape: {original_data.dropna().shape}")

index_2x_data = pd.concat([index_2x, ma_2x], axis=1)
index_2x_data.columns = ['Index_2x', 'MA_2x']
index_2x_data.dropna().to_csv('2x_index.csv')
print(f"2x_index.csv saved with shape: {index_2x_data.dropna().shape}")

index_3x_data = pd.concat([index_3x, ma_3x], axis=1)
index_3x_data.columns = ['Index_3x', 'MA_3x']
index_3x_data.dropna().to_csv('3x_index.csv')
print(f"3x_index.csv saved with shape: {index_3x_data.dropna().shape}")

# Combine all the data into a single dataframe for aligned_data.csv
aligned_data = pd.concat({
    'Price': price,
    'MA_Index': ma_index,
    'Cumulative_2x': index_2x,
    'MA_2x': ma_2x
}, axis=1)

# Flatten the MultiIndex columns
aligned_data.columns = aligned_data.columns.get_level_values(0)

# Print column names to verify the structure
print(f"Aligned Data Columns: {aligned_data.columns}")

aligned_data.dropna().to_csv('aligned_data.csv')
print(f"aligned_data.csv saved with shape: {aligned_data.dropna().shape}")

# Plotting (Log Scale)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot original index with log scale
ax1.plot(price, label=f'{ticker} Price', color='blue')
ax1.plot(ma_index, label=f'{N}-Day MA', color='orange', linestyle='--')
ax1.set_title(f'{ticker} Index Price')
ax1.set_ylabel('Price')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, which="both", ls="--")

# Plot 2x synthetic index with log scale
ax2.plot(index_2x, label='2x Synthetic Index', color='green')
ax2.plot(ma_2x, label=f'{N}-Day MA', color='orange', linestyle='--')
ax2.plot(ma_2x_half, label=f'{N//2}-Day MA', color='purple', linestyle='--')  # Plot n/2 MA
ax2.set_title('2x Leveraged Index (Point-Based)')
ax2.set_ylabel('Synthetic Price')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, which="both", ls="--")

# Plot 3x synthetic index with log scale
ax3.plot(index_3x, label='3x Synthetic Index', color='red')
ax3.plot(ma_3x, label=f'{N}-Day MA', color='orange', linestyle='--')
ax3.set_title('3x Leveraged Index (Point-Based)')
ax3.set_ylabel('Synthetic Price')
ax3.set_xlabel('Date')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig('indices_plot.png')
plt.show()

# Now call the strategy function to compute and plot net ROIC
print("Running strategy simulation script...")
strategy.run_strategy_simulation(aligned_data)

