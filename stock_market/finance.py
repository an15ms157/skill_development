import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

# Function to perform SIP investment
def calculate_sip(stock_prices, sip_amount, interval_days):
    num_investments = len(stock_prices) // interval_days
    investment_dates = [stock_prices.index[i * interval_days] for i in range(num_investments)]
    total_units = 0
    total_cost = 0
    
    for date in investment_dates:
        price = stock_prices.loc[date]
        units = sip_amount / price
        total_units += units
        total_cost += sip_amount
    
    average_price_sip = total_cost / total_units
    final_value = total_units * stock_prices[-1]
    return final_value, average_price_sip, investment_dates

# Function to calculate lump-sum investment
def calculate_lump_sum(stock_prices, investment_amount):
    initial_price = stock_prices[0]
    units = investment_amount / initial_price
    final_value = units * stock_prices[-1]
    return final_value, initial_price

# Main function to compare SIP and lump-sum investments
def compare_investments(ticker, start_date, end_date, sip_amount, interval_days):
    stock_prices = fetch_stock_data(ticker, start_date, end_date)
    
    sip_final_value, average_price_sip, sip_dates = calculate_sip(stock_prices, sip_amount, interval_days)
    lump_sum_final_value, average_price_lump_sum = calculate_lump_sum(stock_prices, sip_amount * (len(stock_prices) // interval_days))
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(stock_prices.index, stock_prices, label=f'{ticker} Stock Price', color='blue')
    plt.axhline(y=average_price_sip, color='orange', linestyle='-.', label=f'Average SIP Price: {average_price_sip:.2f}')
    plt.axhline(y=average_price_lump_sum, color='purple', linestyle='-.', label=f'Average Lump-Sum Price: {average_price_lump_sum:.2f}')
    
    # Plot SIP investment points
    sip_prices = stock_prices[sip_dates]
    plt.scatter(sip_dates, sip_prices, color='green', label='SIP Investment Days', zorder=5)
    
    # Adding text boxes
    plt.text(stock_prices.index[-1], sip_final_value, f'SIP Final Value: {sip_final_value:.2f}', 
             horizontalalignment='left', verticalalignment='bottom', bbox=dict(facecolor='green', alpha=0.5))
    plt.text(stock_prices.index[-1], lump_sum_final_value, f'Lump-Sum Final Value: {lump_sum_final_value:.2f}', 
             horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title(f'Comparison of SIP and Lump-Sum Investments in {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
sip_amount = 300
interval_days = 100

compare_investments(ticker, start_date, end_date, sip_amount, interval_days)

