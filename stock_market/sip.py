import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy_financial as npf
import random
import time
import requests
from requests.exceptions import RequestException

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fetch_sp500_data(start_date, end_date, retries=3, delay=5):
    print("Fetching S&P 500 daily data...")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })

    for attempt in range(retries):
        try:
            sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", timeout=10, session=session)
            if not sp500.empty:
                print("Data fetched successfully.")
                return sp500
            else:
                print(f"Attempt {attempt + 1}: Empty data.")
        except (RequestException, Exception) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    print("Failed to fetch data.")
    return None

# Fetch data
sp500 = fetch_sp500_data("1950-01-01", "2024-12-31")
if sp500 is None or sp500.empty:
    exit("No data fetched. Exiting.")

df = sp500[['Close']].rename(columns={'Close': 'Index'})
df['Date'] = df.index
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Index'] = df['Index'].fillna(method='ffill')
df = df.reset_index(drop=True)

print("Starting 30-Year Random Day SIP Simulation...")

roic_results = []
rolling_years = 30
min_year = df['Year'].min()
max_year = df['Year'].max() - rolling_years + 1

for start_year in range(min_year, max_year + 1):
    end_year = start_year + rolling_years - 1
    df_period = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()

    df_period['YearMonth'] = df_period['Year'].astype(str) + '-' + df_period['Month'].astype(str).str.zfill(2)
    grouped = df_period.groupby('YearMonth')

    investment_dates = []
    investment_prices = []

    for _, group in grouped:
        if not group.empty:
            random_idx = random.randint(0, len(group) - 1)
            investment_dates.append(group.iloc[random_idx]['Date'])
            investment_prices.append(group.iloc[random_idx]['Index'])

    if not investment_prices:
        continue

    units = 1 / np.array(investment_prices)
    total_units = np.sum(units)
    
    # Fix: use .iloc[-1] directly (already a scalar)
    final_price = df_period['Index'].iloc[-1]
    final_value = total_units * final_price

    total_invested = len(investment_prices)
    net_return = (final_value - total_invested) / total_invested * 100  # Net ROIC %

    roic_results.append({
        'StartYear': start_year,
        'EndYear': end_year,
        'NetROIC': net_return,
        'StartDate': df_period['Date'].iloc[0],
        'EndDate': df_period['Date'].iloc[-1],
        'IndexValues': df_period[['Date', 'Index']]
    })

# Prepare for plotting
results_df = pd.DataFrame(roic_results)
start_years = results_df['StartYear']
net_roics = results_df['NetROIC']

# Save ROIC results to Excel
excel_df = results_df[['StartYear', 'EndYear', 'NetROIC']]
excel_df.to_excel("30yr_random_day_sip.xlsx", index=False)
print("Excel file saved as 30yr_random_day_sip.xlsx")

# Create figure with subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# Subplot 1: S&P 500 Index during each 30-year period
for record in roic_results:
    index_df = record['IndexValues']
    axs[0].plot(index_df['Date'], index_df['Index'], alpha=0.3)

axs[0].set_title("S&P 500 Index over Rolling 30-Year Windows (1950–2024)")
axs[0].set_ylabel("Index Value")
axs[0].set_yscale("log")
axs[0].grid(True)

# Subplot 2: Net ROIC bar graph
axs[1].bar(start_years, net_roics, color='teal')
axs[1].set_title("Net ROIC of Random Day SIP Strategy (30-Year Windows)")
axs[1].set_xlabel("Start Year of 30-Year Window")
axs[1].set_ylabel("Net ROIC (%)")
axs[1].grid(True, axis='y')

# Layout and save
plt.tight_layout()
plt.savefig("30yr_random_day_sip.png")
print("Plot saved as 30yr_random_day_sip.png")

