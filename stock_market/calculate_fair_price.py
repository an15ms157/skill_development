import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_fair_value_over_time(ticker, discount_rate, growth_rate, years=10):
    # Download stock data (daily prices)
    stock = yf.Ticker(ticker)
    historical_prices = stock.history(period=f"{years}y", interval="1d")  # Daily data for the last 'years' years
    
    # Retrieve cashflow data
    try:
        cash_flows = stock.cashflow.loc['Operating Cash Flow']
    except KeyError:
        raise ValueError("Operating Cash Flow data not available for this stock.")
    
    # Ensure sufficient cash flow data is available
    if len(cash_flows) < years:
        raise ValueError("Not enough historical cash flow data to calculate fair value.")
    
    # Prepare fair value calculation
    fair_values = []
    dates = []
    
    # Iterate over available cash flows
    for idx, cash_flow in enumerate(cash_flows):
        if np.isnan(cash_flow) or idx >= years:  # Skip if cash flow is NaN or exceeds required years
            continue
        
        # Estimate future cash flows
        future_cash_flows = [cash_flow * (1 + growth_rate) ** j for j in range(1, 6)]
        # Calculate present value of future cash flows
        present_values = [cf / (1 + discount_rate) ** j for j, cf in enumerate(future_cash_flows, start=1)]
        intrinsic_value = sum(present_values)
        
        # Get shares outstanding
        shares_outstanding = stock.info.get('sharesOutstanding', 1)  # Default to 1 to avoid division by zero
        if shares_outstanding == 0:
            continue
        
        # Calculate fair price
        fair_price = intrinsic_value / shares_outstanding
        fair_values.append(fair_price)
        dates.append(cash_flows.index[idx].strftime('%Y-%m-%d'))  # Convert timestamp to string
    
    # Create DataFrame for fair value
    fair_values_df = pd.DataFrame({'Date': pd.to_datetime(dates), 'FairValue': fair_values})
    fair_values_df.set_index('Date', inplace=True)
    
    # Align historical prices with fair values
    historical_prices['Date'] = historical_prices.index.date
    stock_prices = historical_prices.set_index('Date')['Close']
    
    # Merge the stock prices with fair value data (reindex to daily)
    combined = stock_prices.to_frame().join(fair_values_df, how='left').fillna(method='ffill')  # Forward fill fair values
    
    return combined

def plot_fair_value_evolution(ticker, combined):
    plt.figure(figsize=(14, 7))
    plt.plot(combined.index, combined['FairValue'], label="Fair Value", linestyle="-", color="blue", linewidth=2)
    plt.plot(combined.index, combined['Close'], label="Stock Price", linestyle="-", color="green", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"Fair Value Evolution vs Stock Price for {ticker}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
ticker = "MSFT"  # Replace with your stock ticker
discount_rate = 0.1  # 10% discount rate
growth_rate = 0.05  # 5% growth rate
years = 5  # Time horizon for the plot

try:
    combined = calculate_fair_value_over_time(ticker, discount_rate, growth_rate, years)
    plot_fair_value_evolution(ticker, combined)
except ValueError as e:
    print(f"Error: {e}")

