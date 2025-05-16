import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define indices and their Yahoo Finance tickers
tickers = {
    'US': '^GSPC',        # S&P 500 (USD)
    'Germany': '^GDAXI',  # DAX (EUR, DEM pre-1999)
    'China': '000001.SS', # Shanghai Composite (CNY)
    'India': '^BSESN',    # BSE Sensex (INR)
    'Japan': '^N225'      # Nikkei 225 (JPY)
}

# Define exchange rate tickers (price of USD in foreign currency, e.g., EUR/USD)
exchange_rates = {
    'Germany': 'EURUSD=X',  # EUR/USD (DEM/USD approximated pre-1999)
    'China': 'CNYUSD=X',    # CNY/USD
    'India': 'INRUSD=X',    # INR/USD
    'Japan': 'JPYUSD=X'     # JPY/USD
    # US: No conversion needed
}

# Download historical monthly data from 1970 to present
data = {}
for country, ticker in tickers.items():
    df = yf.download(ticker, start='1970-01-01', interval='1mo', progress=False)
    df = df[['Close']].rename(columns={'Close': country})
    data[country] = df

# Download exchange rates
fx_data = {}
for country, fx_ticker in exchange_rates.items():
    fx = yf.download(fx_ticker, start='1970-01-01', interval='1mo', progress=False)
    fx = fx[['Close']].rename(columns={'Close': f'{country}_USD'})
    fx_data[country] = fx

# Convert indices to USD
usd_data = {}
for country in tickers:
    if country == 'US':
        usd_data[country] = data[country]  # No conversion needed
    else:
        # Merge index and exchange rate data
        df = data[country].join(fx_data[country])
        # Convert to USD: Index price * (USD per foreign currency)
        df[country] = df[country] * df[f'{country}_USD']
        usd_data[country] = df[[country]]

# Combine all USD-converted data into a single DataFrame
combined_df = pd.concat([usd_data[country] for country in tickers], axis=1)

# Fill missing values
combined_df.fillna(method='ffill', inplace=True)
combined_df.fillna(method='bfill', inplace=True)

# Debug: Check data
print(f"USD Combined data shape: {combined_df.shape}")
print(f"USD Combined data index range: {combined_df.index[0]} to {combined_df.index[-1]}")
print(f"USD Combined data head:\n{combined_df.head()}")

# Normalize each index to start at 100
normalized_df = combined_df / combined_df.iloc[0] * 100

# Define dynamic allocation function
def get_weights(date):
    if date.year < 1990:
        return {'US': 0.4, 'Germany': 0.2, 'Japan': 0.4, 'China': 0.0, 'India': 0.0}
    elif date.year < 1995:
        return {'US': 0.4, 'Germany': 0.2, 'Japan': 0.3, 'China': 0.05, 'India': 0.05}
    elif date.year < 1998:
        return {'US': 0.4, 'Germany': 0.2, 'Japan': 0.1, 'China': 0.15, 'India': 0.15}
    elif date.year < 2000:
        return {'US': 0.4, 'Germany': 0.2, 'Japan': 0.05, 'China': 0.15, 'India': 0.2}
    else:
        return {'US': 0.4, 'Germany': 0.2, 'China': 0.2, 'India': 0.2, 'Japan': 0.0}

# Calculate portfolio value over time
portfolio_values = []
for date, row in normalized_df.iterrows():
    weights = get_weights(date)
    try:
        value = sum(float(row[country]) * weight for country, weight in weights.items() if country in row.index and not pd.isna(row[country]))
        portfolio_values.append(value)
    except Exception as e:
        print(f"Error calculating portfolio value for date {date}: {e}")
        portfolio_values.append(np.nan)

# Create a DataFrame for portfolio values
portfolio_df = pd.DataFrame({
    'Portfolio': portfolio_values
}, index=normalized_df.index)

# Drop rows with invalid portfolio values
portfolio_df.dropna(inplace=True)

# Normalize portfolio to 100 at the start
portfolio_df["Portfolio"] = portfolio_df["Portfolio"] / portfolio_df["Portfolio"].iloc[0] * 100

# Debugging output
print("Normalized USD DataFrame:")
print(normalized_df.head())
print("Portfolio USD DataFrame:")
print(portfolio_df.head())

# Plot 1: Dynamic Portfolio from Construction Time
plt.figure(figsize=(12, 6))
plt.plot(portfolio_df.index, portfolio_df["Portfolio"], label='Dynamic Global Portfolio (USD)', color='blue', linewidth=2)
plt.title('Dynamic Global Portfolio in USD (From Construction Time)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value (Percentage of Initial)', fontsize=12)
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dynamic_portfolio_usd_from_construction.png")
plt.close()
print("Chart saved as 'dynamic_portfolio_usd_from_construction.png'")

# Plot 2: Constituent Indices and Dynamic Portfolio (Starting from a Given Date)
start_date = '1997-01-01'

# Filter data to start from the given date
constituent_df = combined_df.loc[start_date:]
portfolio_df_filtered = portfolio_df.loc[start_date:]

# Normalize all indices and the portfolio to 100 at the start date
constituent_df = constituent_df / constituent_df.iloc[0] * 100
portfolio_df_filtered["Portfolio"] = portfolio_df_filtered["Portfolio"] / portfolio_df_filtered["Portfolio"].iloc[0] * 100

plt.figure(figsize=(12, 6))
# Plot the dynamic portfolio
plt.plot(portfolio_df_filtered.index, portfolio_df_filtered["Portfolio"],
         label='Dynamic Global Portfolio (USD)', color='blue', linewidth=2)

# Plot each constituent index
for column in constituent_df.columns:
    plt.plot(constituent_df.index, constituent_df[column], label=column, linestyle='--', alpha=0.7)

plt.title(f'Constituent Indices and Dynamic Portfolio in USD (From {start_date})', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value (Percentage of Initial)', fontsize=12)
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"constituent_indices_with_dynamic_index_usd_from_{start_date.replace('-', '_')}.png")
plt.close()
print(f"Chart saved as 'constituent_indices_with_dynamic_index_usd_from_{start_date.replace('-', '_')}.png'")
