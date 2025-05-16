import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
TICKER = "BAJFINANCE.NS"
WACC = 0.07  # Weighted Average Cost of Capital (7%)
GROWTH_RATE = 0.05  # Adjusted Growth rate for growth-adjusted EPV (5%)
START_YEAR = 1996  # Earliest year in Excel data
END_YEAR = 2024   # Latest year in Excel data
YEARS_FOR_AVERAGE = 5  # Number of years to average Net Income
STOCK_START_YEAR = 1990  # Start stock price data from 2000 (post-IPO)

# File paths
income_file = f"data/{TICKER}/{TICKER} Income Statement (Annual) - Discounting Cash Flows.xlsx"
balance_file = f"data/{TICKER}/{TICKER} Balance Sheet Statement (Annual) - Discounting Cash Flows.xlsx"

# Read Excel files
try:
    income_df = pd.read_excel(income_file, sheet_name=0, header=0, index_col=0)
    balance_df = pd.read_excel(balance_file, sheet_name=0, header=0, index_col=0)
    print(f"Successfully read Excel files for {TICKER}")
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit(1)
except Exception as e:
    print(f"Error reading Excel files for {TICKER}: {e}")
    exit(1)

# Clean index and column names
for df in [income_df, balance_df]:
    df.index = df.index.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
    df.columns = [col if col == 'LTM' else col.split(' ')[0].split('-')[0] for col in df.columns]

# Define row labels
net_income_row = "Net Income"
shares_row = "Diluted Weighted Average Shares Outstanding"
net_debt_row = "Net Debt"

# Robust row label matching
def find_row_label(df, target_label):
    for row in df.index:
        if target_label.lower() in row.lower():
            return row
    return None

# Validate required rows
net_income_label = find_row_label(income_df, net_income_row)
shares_label = find_row_label(income_df, shares_row)
net_debt_label = find_row_label(balance_df, net_debt_row)

if not all([net_income_label, shares_label, net_debt_label]):
    print("Error: Missing required rows in Excel files")
    print(f"Net Income: {net_income_label}, Shares: {shares_label}, Net Debt: {net_debt_label}")
    exit(1)

print(f"Found rows: Net Income='{net_income_label}', Shares='{shares_label}', Net Debt='{net_debt_label}'")

# Extract data
available_years = [col for col in income_df.columns if col.isdigit() and START_YEAR <= int(col) <= END_YEAR]
if not available_years:
    print(f"Error: No valid years found in Income Statement columns")
    exit(1)

# Convert to numeric, handling strings
def convert_to_numeric(series):
    result = series.copy()
    for idx in result.index:
        value = result[idx]
        if isinstance(value, str):
            value = value.replace(',', '').replace('$', '').replace(' million', '').strip()
            try:
                result[idx] = float(value)
            except ValueError:
                result[idx] = np.nan
        elif isinstance(value, (int, float)):
            result[idx] = float(value)
        else:
            result[idx] = np.nan
    return pd.to_numeric(result, errors='coerce')

net_income = convert_to_numeric(income_df.loc[net_income_label, available_years])
shares = convert_to_numeric(income_df.loc[shares_label, available_years])
net_debt = convert_to_numeric(balance_df.loc[net_debt_label, available_years])

# Debug: Print extracted data
print("\nExtracted Data:")
print("Net Income ($M):")
print(net_income)
print("Shares Outstanding (M):")
print(shares)
print("Net Debt ($M):")
print(net_debt)

# Calculate historical EPV
def calculate_epv(year, net_income, net_debt, shares):
    # Get 5-year average Net Income
    start_year = year - YEARS_FOR_AVERAGE + 1
    years_range = [str(y) for y in range(start_year, year + 1) if str(y) in net_income.index]
    if len(years_range) < YEARS_FOR_AVERAGE:
        print(f"Skipping year {year}: Not enough data ({len(years_range)} years available)")
        return np.nan, np.nan  # Not enough data
    avg_net_income = net_income[years_range].mean()
    print(f"Year {year}: Avg Net Income = {avg_net_income:.2f} million")
    
    if np.isnan(avg_net_income) or np.isnan(net_debt) or np.isnan(shares) or shares <= 0:
        print(f"Skipping year {year}: Invalid data (Avg Net Income={avg_net_income}, Net Debt={net_debt}, Shares={shares})")
        return np.nan, np.nan
    
    # Base EPV
    base_epv = (avg_net_income / WACC - net_debt) / shares
    
    # Growth-Adjusted EPV
    if WACC > GROWTH_RATE:
        growth_epv = (avg_net_income * (1 + GROWTH_RATE) / (WACC - GROWTH_RATE) - net_debt) / shares
    else:
        growth_epv = np.nan
    
    print(f"Year {year}: Base EPV = {base_epv:.2f}, Growth EPV = {growth_epv:.2f}")
    return base_epv, growth_epv

# Compute EPV starting from STOCK_START_YEAR (2000)
base_epv = []
growth_epv = []
epv_years = []
for year in range(max(START_YEAR + YEARS_FOR_AVERAGE - 1, STOCK_START_YEAR), END_YEAR + 1):  # Start from 2000
    if str(year) in net_income.index and str(year) in net_debt.index and str(year) in shares.index:
        base, growth = calculate_epv(year, net_income, net_debt[str(year)], shares[str(year)])
        base_epv.append(base)
        growth_epv.append(growth)
        epv_years.append(datetime(year, 12, 31))  # Convert year to datetime (Dec 31)

# Debug: Print EPV data
print("\nEPV Data:")
print(f"Years: {len(epv_years)} entries")
print(f"Base EPV: {len(base_epv)} entries")
print(f"Growth EPV: {len(growth_epv)} entries")
for y, b, g in zip(epv_years, base_epv, growth_epv):
    print(f"Year {y.year}: Base EPV = {b:.2f}, Growth EPV = {g:.2f}")

# Fetch stock price data from yfinance
end_date = datetime(2025, 4, 26)
start_date = datetime(STOCK_START_YEAR, 1, 1)
stock_data = yf.download(TICKER, start=start_date, end=end_date, interval='1d')

# Resample to weekly (using last trading day of the week)
stock_data_weekly = stock_data['Close'].resample('W-FRI').last()

# Calculate 50-day SMA on daily data
stock_data['50_SMA'] = stock_data['Close'].rolling(window=50).mean()

# Resample 50-day SMA to weekly
sma_weekly = stock_data['50_SMA'].resample('W-FRI').last()

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot weekly stock prices and 50-day SMA (left y-axis)
ax1.plot(stock_data_weekly.index, stock_data_weekly, label='Weekly Stock Price', color='blue')
ax1.plot(sma_weekly.index, sma_weekly, label='50-day SMA', color='orange', linestyle='--')
ax1.plot(epv_years, base_epv, label='Base EPV (No Growth)', color='green', marker='o', linestyle='-')
ax1.plot(epv_years, growth_epv, label='Growth EPV (5%)', color='red', marker='o', linestyle='-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price / EPV (USD)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xlim(start_date, end_date)  # Set x-axis limits to match stock data
ax1.set_yscale('log')  # Log scale to handle wide range of values
ax1.set_ylim(10, 10000)  # Adjust y-axis limits to ensure visibility
ax1.legend(loc='upper left')

# Title and grid
plt.title(f'AMZN Historical EPV and Weekly Stock Price with 50-day SMA (2000–2025)')
ax1.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig(f'{TICKER}_epv_stock_plot.png')
plt.close()

print(f"Plot saved as '{TICKER}_epv_stock_plot.png'")