import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import requests
from requests.exceptions import RequestException
from RandomDay30yr_sip import perform_sip_simulation as perform_random_sip
from FixedDay30yr_sip import perform_sip_simulation as perform_fixed_sip
from WorstDay30yr_sip import perform_sip_simulation as perform_worst_sip
from BestDay30yr_sip import perform_sip_simulation as perform_best_sip
from Below200DMA30yr_sip import perform_sip_simulation as perform_below_dma_sip

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICKER = "^GSPC"  # Ticker for Shanghai Composite Index
StartDate="1950-01-01"
EndDate="2024-12-31"

# 🔧 SELECT STRATEGIES TO RUN
RUN_STRATEGIES = {
    'random': True,
    'fixed': True,
    'worst': True,
    'best': True,
    'below_200dma': True
}

def fetch_sp500_data(start_date, end_date, retries=3, delay=5):
    print("Fetching S&P 500 daily data...")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })

    for attempt in range(retries):
        try:
            sp500 = yf.download(TICKER, start=start_date, end=end_date, interval="1d", timeout=10, session=session)
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

sp500 = fetch_sp500_data(StartDate, EndDate)
if sp500 is None or sp500.empty:
    exit("No data fetched. Exiting.")

# Prepare the DataFrame
df = sp500[['Close']].rename(columns={'Close': 'Index'})
df['Date'] = df.index
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Index'] = df['Index'].fillna(method='ffill')
df = df.reset_index(drop=True)

# Store results
roic_data = {}
start_years = None

# --- STRATEGY RUNS ---
if RUN_STRATEGIES['random']:
    print("Starting 30-Year Random Day SIP Simulation...")
    random_results = perform_random_sip(df)
    if random_results:
        roic_data['random'] = pd.DataFrame(random_results)
        start_years = roic_data['random']['StartYear']
    else:
        print("No investments made for Random Day strategy.")

if RUN_STRATEGIES['fixed']:
    print("Starting 30-Year Fixed Day SIP Simulation...")
    fixed_results = perform_fixed_sip(df)
    if fixed_results:
        roic_data['fixed'] = pd.DataFrame(fixed_results)
        if start_years is None:
            start_years = roic_data['fixed']['StartYear']
    else:
        print("No investments made for Fixed Day strategy.")

if RUN_STRATEGIES['worst']:
    print("Starting 30-Year Worst Day SIP Simulation...")
    worst_results = perform_worst_sip(df)
    if worst_results:
        roic_data['worst'] = pd.DataFrame(worst_results)
        if start_years is None:
            start_years = roic_data['worst']['StartYear']
    else:
        print("No investments made for Worst Day strategy.")

if RUN_STRATEGIES['best']:
    print("Starting 30-Year Best Day SIP Simulation...")
    best_results = perform_best_sip(df)
    if best_results:
        roic_data['best'] = pd.DataFrame(best_results)
        if start_years is None:
            start_years = roic_data['best']['StartYear']
    else:
        print("No investments made for Best Day strategy.")

if RUN_STRATEGIES['below_200dma']:
    print("Starting 30-Year Below 200-Day DMA SIP Simulation...")
    below_dma_results = perform_below_dma_sip(df)
    if below_dma_results:
        roic_data['below_200dma'] = pd.DataFrame(below_dma_results)
        if start_years is None:
            start_years = roic_data['below_200dma']['StartYear']
    else:
        print("No investments made for Below 200-DMA strategy.")
        exit("No data to plot. Exiting.")

# Check if any strategies produced results
if not roic_data:
    exit("No strategies produced results. Exiting.")

# --- COMBINE RESULTS TO CSV ---
combined_roic_df = pd.DataFrame({'Start_Year': start_years})
if 'random' in roic_data:
    combined_roic_df['Net_ROIC_Random_Percent'] = roic_data['random']['NetROIC']
if 'fixed' in roic_data:
    combined_roic_df['Net_ROIC_Fixed_Percent'] = roic_data['fixed']['NetROIC']
if 'worst' in roic_data:
    combined_roic_df['Net_ROIC_Worst_Percent'] = roic_data['worst']['NetROIC']
if 'best' in roic_data:
    combined_roic_df['Net_ROIC_Best_Percent'] = roic_data['best']['NetROIC']
if 'below_200dma' in roic_data:
    combined_roic_df['Net_ROIC_Below_200DMA_Percent'] = roic_data['below_200dma']['NetROIC']

combined_roic_df.to_csv('roic_comparison_all_strategies.csv', index=False)
print("ROIC comparison data saved to roic_comparison_all_strategies.csv")

# --- PLOT ---
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# Subplot 1: Index Chart (use first available strategy)
for key in roic_data:
    for record in roic_data[key].itertuples():
        axs[0].plot(record.IndexValues['Date'], record.IndexValues['Index'], alpha=0.3)
    break

axs[0].set_title(TICKER + " Index over Rolling 30-Year Windows (1950–2024)")
axs[0].set_ylabel("Index Value")
axs[0].set_yscale("log")
axs[0].grid(True)

# Subplot 2: Bar Plot for ROIC
bar_width = 0.15
x = np.arange(len(start_years))
offset = -((len(roic_data) - 1) / 2) * bar_width

colors = {
    'random': 'teal',
    'fixed': 'coral',
    'worst': 'purple',
    'best': 'green',
    'below_200dma': 'blue'
}

labels = {
    'random': 'Random Day SIP',
    'fixed': 'Fixed Day SIP',
    'worst': 'Worst Day SIP',
    'best': 'Best Day SIP',
    'below_200dma': 'Below 200-DMA SIP'
}

for i, (key, df) in enumerate(roic_data.items()):
    axs[1].bar(x + offset + i * bar_width, df['NetROIC'], bar_width, label=labels[key], color=colors[key])

axs[1].set_title("Net ROIC: SIP Strategies Over 30-Year Windows")
axs[1].set_xlabel("Start Year of 30-Year Window")
axs[1].set_ylabel("Net ROIC (%)")
axs[1].set_xticks(x)
axs[1].set_xticklabels(start_years, rotation=45)
axs[1].legend()
axs[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig("30yr_selected_strategies_sip_subplot.png")
print("Plot saved as 30yr_selected_strategies_sip_subplot.png")
