import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import sip_config

def calculate_sip_returns(input_folder):
    """
    Calculate returns for SIP investment strategy.
    """
    start_date = pd.to_datetime(sip_config["start_date"])
    monthly_investment = sip_config["monthly_investment"]
    weights = sip_config["weights"]
    
    # Read all USD index data
    index_data = {}
    for country in weights.keys():
        file_path = os.path.join(input_folder, f"{country}_usd_index_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df.iloc[:, 0])
            df.set_index('Date', inplace=True)
            df = df.iloc[:, 0]  # Keep only the index values
            index_data[country] = df

    # Create a date range for monthly investments
    all_dates = pd.date_range(start=start_date, end=max(df.index for df in index_data.values()), freq='M')
    
    # Initialize portfolio tracking
    portfolio = pd.DataFrame(index=all_dates)
    total_investment = 0
    units_held = {country: 0 for country in weights.keys()}
    
    # Calculate units purchased each month and portfolio value
    for date in all_dates:
        # For each country, calculate units that can be bought with the weighted investment
        for country, weight in weights.items():
            investment_amount = monthly_investment * weight
            total_investment += investment_amount
            
            # Find the closest date in index data (same or next available date)
            index_series = index_data[country]
            investment_date = index_series.index[index_series.index >= date][0]
            price = index_series[investment_date]
            
            # Calculate and add new units
            new_units = investment_amount / price
            units_held[country] += new_units
            
            # Calculate value of holdings for this country
            portfolio.loc[date, f'{country}_Value'] = units_held[country] * price
    
    # Calculate total portfolio value and ROIC for each date
    portfolio['Total_Value'] = portfolio.sum(axis=1)
    portfolio['Total_Investment'] = range(1, len(portfolio) + 1)
    portfolio['Total_Investment'] = portfolio['Total_Investment'] * monthly_investment
    portfolio['ROIC'] = (portfolio['Total_Value'] / portfolio['Total_Investment'] - 1) * 100
    
    return portfolio

def plot_sip_performance(portfolio, output_folder):
    """
    Plot SIP strategy performance including portfolio value and ROIC.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot total portfolio value
    ax1.plot(portfolio.index, portfolio['Total_Value'], label='Portfolio Value', linewidth=2)
    ax1.plot(portfolio.index, portfolio['Total_Investment'], label='Total Investment', linewidth=2, linestyle='--')
    ax1.set_title('SIP Strategy: Portfolio Value vs Total Investment', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('USD', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot ROIC
    ax2.plot(portfolio.index, portfolio['ROIC'], label='Return on Invested Capital (%)', linewidth=2, color='green')
    ax2.set_title('SIP Strategy: Return on Invested Capital', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('ROIC (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_folder, exist_ok=True)
    plot_file = os.path.join(output_folder, "sip_strategy_performance.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    print(f"Final Portfolio Value: ${portfolio['Total_Value'].iloc[-1]:.2f}")
    print(f"Total Investment: ${portfolio['Total_Investment'].iloc[-1]:.2f}")
    print(f"Overall ROIC: {portfolio['ROIC'].iloc[-1]:.2f}%")
    print(f"Performance plot saved as {plot_file}")

if __name__ == "__main__":
    input_folder = "usd_index_data"
    output_folder = "plots"
    
    portfolio = calculate_sip_returns(input_folder)
    plot_sip_performance(portfolio, output_folder)