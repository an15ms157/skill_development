# This module implements a moving-average-based investment strategy for a 2x leveraged index.
# It simulates monthly investments, dynamically switching between investing and holding cash based on long-term and short-term moving averages.
# The main function, compute_strategy, returns either the final portfolio value or the return on invested capital (ROIC) for a given period.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

n = 10  # Period in years, adjust as needed for your analysis

def compute_strategy(start_idx, end_idx, data, pfolio=0):
    portfolio_value = 0
    invested_capital = 0
    units_2x = 0
    cash = 0
    ma_n = data['Cumulative_2x'].rolling(window=n*252).mean()
    ma_n_half = data['Cumulative_2x'].rolling(window=(n//2)*252).mean()
    monthly_invest_dates = pd.date_range(data.index[start_idx], data.index[end_idx], freq='MS')
    for date in data.index[start_idx:end_idx + 1]:
        if pd.isna(data['MA_2x'][date]):
            continue
        cum_2x = data['Cumulative_2x'][date]
        ma_2x = data['MA_2x'][date]
        portfolio_value = units_2x * cum_2x + cash
        if cum_2x < ma_n[date] and units_2x > 0:
            cash = portfolio_value
            units_2x = 0
            portfolio_value = cash
        if ma_n_half[date] < cum_2x < ma_n[date]:
            amount_to_invest = cash + 1 if cash > 0 else 1
            units_2x += amount_to_invest / cum_2x
            invested_capital += 1
            cash = 0
            portfolio_value = units_2x * cum_2x
        else:
            invested_capital += 1
    final_roic = ((portfolio_value - invested_capital) / invested_capital)
    if pfolio:
        return portfolio_value
    else:
        return final_roic

def compute_and_plot_net_roic_and_value(aligned_data):
    aligned_data = aligned_data.dropna(subset=['MA_2x'])
    window_days = n * 252
    roic_values = []
    portfolio_values = []
    start_years = []
    for start_idx in range(0, len(aligned_data) - window_days + 1, 252):
        end_idx = min(start_idx + window_days - 1, len(aligned_data) - 1)
        roic = compute_strategy(start_idx, end_idx, aligned_data, pfolio=0)
        portfolio_value = compute_strategy(start_idx, end_idx, aligned_data, pfolio=1)
        roic_values.append(roic)
        portfolio_values.append(portfolio_value)
        start_year = aligned_data.index[start_idx].year
        start_years.append(start_year)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    ax1.plot(start_years, roic_values, label='Net ROIC', color='purple', marker='o')
    ax1.set_title('Net ROIC vs. Starting Year for Strategy (^GSPC)')
    ax1.set_xlabel('Start Year')
    ax1.set_ylabel('Net ROIC (xN)')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(start_years, portfolio_values, label='Net Portfolio Value', color='green', marker='x')
    ax2.set_title('Net Portfolio Value vs. Starting Year for Strategy (^GSPC)')
    ax2.set_xlabel('Start Year')
    ax2.set_ylabel('Net Portfolio Value ($)')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('net_roic_and_portfolio_value_plot.png')
    plt.show()
    return roic_values, portfolio_values, start_years

def run_strategy_simulation(aligned_data):
    compute_and_plot_net_roic_and_value(aligned_data)

