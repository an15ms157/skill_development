import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta, timezone

# Function to calculate correlation within a specified date range
def calculate_correlation(main_ticker, other_ticker, start_date, end_date):
    main_stock = yf.Ticker(main_ticker)
    other_stock = yf.Ticker(other_ticker)

    main_data = main_stock.history(start=start_date, end=end_date)
    other_data = other_stock.history(start=start_date, end=end_date)

    # Ensure both dataframes have the same index (date) to calculate correlation
    main_data = main_data['Close']
    other_data = other_data['Close']

    # Drop NaN values
    main_data = main_data.dropna()
    other_data = other_data.dropna()

    correlation = main_data.corr(other_data)
    return correlation

# Read the stock information from an Excel sheet
stock_info = pd.read_excel('stock_info.xlsx')

# Create a Ticker object to fetch stock data
main_ticker = yf.Ticker(stock_info['Ticker'].iloc[0])  # Assuming all rows have the same ticker

# Define the buy dates and corresponding buy prices from the Excel sheet
buy_dates = pd.to_datetime(stock_info['Buy Date']).dt.tz_localize('UTC').tolist()
buy_prices = stock_info['Buy Price'].tolist()

# Replace 'year', 'month', and 'day' with your desired date
specific_date = datetime(2023, 8, 4, tzinfo=timezone.utc)

# Calculate the length between the specific date and the first buy date
num_days = (specific_date - buy_dates[0]).days

# Calculate the start date based on the specific date
start_date = specific_date - timedelta(days=num_days)

# Fetch data for the main ticker
data_main_ticker = main_ticker.history(start=start_date, end=specific_date)

# Extract the date and closing price
dates_main_ticker = data_main_ticker.index
closing_prices_main_ticker = data_main_ticker['Close']

# Initialize lists to store the indices of the nearest available dates
buy_indices = []

# Find the indices of the nearest available dates for the specified buy dates
for date in buy_dates:
    nearest_date = min(dates_main_ticker, key=lambda x: abs(x - date))
    buy_indices.append(dates_main_ticker.get_loc(nearest_date))

# Calculate the specific day's price for the main ticker
specific_day_price_main_ticker = closing_prices_main_ticker[-1]

# Print the main ticker information
print(f'{stock_info["Ticker"].iloc[0]}:')
print(f'Specific Date: {specific_date}')
print(f'Specific Day\'s Price: ${specific_day_price_main_ticker:.2f}')

# Specify the symbol of the other ticker
other_ticker_symbol = 'SOL-USD'  # Replace with your desired ticker symbol

# Calculate correlation within the specified date range
correlation = calculate_correlation(stock_info['Ticker'].iloc[0], other_ticker_symbol, start_date, specific_date)

print(f'Correlation between {stock_info["Ticker"].iloc[0]} and {other_ticker_symbol}: {correlation:.2f}')

# Fetch data for the other ticker
other_ticker = yf.Ticker(other_ticker_symbol)
data_other_ticker = other_ticker.history(start=start_date, end=specific_date)

# Extract the date and closing price for the other ticker
dates_other_ticker = data_other_ticker.index
closing_prices_other_ticker = data_other_ticker['Close']

# Convert prices to percentage change from the start date
percentage_change_main_ticker = (closing_prices_main_ticker / closing_prices_main_ticker.iloc[0] - 1) * 100
percentage_change_other_ticker = (closing_prices_other_ticker / closing_prices_other_ticker.iloc[0] - 1) * 100

# Plotting side by side with two subplots for the second plot
plt.figure(figsize=(15, 6))

# Plot for the main ticker
plt.subplot(1, 2, 1)
plt.plot(dates_main_ticker, closing_prices_main_ticker, label=f'{stock_info["Ticker"].iloc[0]} Closing Price', color='b')
plt.scatter(dates_main_ticker[buy_indices], buy_prices, color='g', marker='o', label='Buy Dates', s=200)
plt.title(f'{stock_info["Ticker"].iloc[0]} Closing Price Over the Specified Period')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()

# Plot for the other ticker (two subplots)
plt.subplot(1, 2, 2)
plt.plot(dates_other_ticker, percentage_change_other_ticker, label=f'{other_ticker_symbol} Closing Price', color='r')
plt.plot(dates_main_ticker, percentage_change_main_ticker, label=f'{stock_info["Ticker"].iloc[0]} Closing Price', color='b', linestyle='dashed')
#plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.title(f'{other_ticker_symbol} and {stock_info["Ticker"].iloc[0]} Closing Prices Over the Specified Period (Percentage Change)')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.grid(True)
plt.tight_layout()  # Ensures that the subplots fit nicely
plt.show()

