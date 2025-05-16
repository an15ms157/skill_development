
# This script computes and plots the annual growth of a 
# portfolio using a given strategy and compares it to a 2x leveraged index.
# The strategy invests $1 each month in a 2x leveraged index, using two moving averages (n-year and n/2-year) as signals:
# - If the 2x index falls below the long-term MA, all holdings are sold and moved to cash.
# - If the 2x index is between the short-term and long-term MAs, all available cash plus $1 is invested.
# - Otherwise, cash is held and $1 is added to available cash.
# The approach aims to grow the portfolio while managing risk using moving average signals.
# See compute_strategy in strategy.py for details.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import strategy

def compute_portfolio_growth(aligned_data):
    if not isinstance(aligned_data.index, pd.DatetimeIndex):
        aligned_data.index = pd.to_datetime(aligned_data.index)
    aligned_data = aligned_data.dropna(subset=['MA_2x'])
    if 'MA_2x' not in aligned_data.columns or 'Cumulative_2x' not in aligned_data.columns:
        raise ValueError("Required columns 'MA_2x' or 'Cumulative_2x' are missing from the data.")
    portfolio_values = []
    start_years = []
    window_days = 252
    for start_idx in range(0, len(aligned_data) - window_days + 1, window_days):
        end_idx = min(start_idx + window_days - 1, len(aligned_data) - 1)
        portfolio_value = strategy.compute_strategy(start_idx, end_idx, aligned_data, pfolio=1)
        portfolio_values.append(portfolio_value)
        start_year = aligned_data.index[start_idx].year
        start_years.append(start_year)
    print("Portfolio Values: ", portfolio_values)
    print("Start Years: ", start_years)
    return portfolio_values, start_years

def plot_portfolio_growth(aligned_data):
    portfolio_values, start_years = compute_portfolio_growth(aligned_data)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    bins = np.linspace(min(portfolio_values), max(portfolio_values), 15)
    ax1.hist(portfolio_values, bins=bins, color='purple', edgecolor='black')
    ax1.set_title('Portfolio Value per Year')
    ax1.set_xlabel('Portfolio Value ($)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True)
    ax2.plot(aligned_data['Cumulative_2x'], label='2x Leveraged Index', color='green')
    ax2.set_title('2x Leveraged Index')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Value ($)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('portfolio_growth_histogram.png')
    plt.show()

aligned_data = pd.read_csv('aligned_data.csv', index_col=0, parse_dates=True)
if aligned_data.empty:
    raise ValueError("The input data is empty. Please provide valid data.")
plot_portfolio_growth(aligned_data)