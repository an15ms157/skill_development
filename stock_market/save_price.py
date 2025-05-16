import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to fetch Bitcoin price data and save to Excel
def fetch_and_fit_btc_prices(ticker="BTC-USD", start_date="2009-01-01", end_date="2023-12-31", file_path="btc_prices.xlsx"):
    # Fetch historical data
    btc_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Save data to Excel
    btc_data.to_excel(file_path)
    print(f"Bitcoin price data saved to {file_path}")
    
    # Use the 'Close' prices for fitting
    btc_data['Date'] = btc_data.index
    btc_data['Days'] = (btc_data['Date'] - btc_data['Date'].min()).dt.days
    prices = btc_data['Close'].values
    days = btc_data['Days'].values

    # Hill function
    def hill_func(x, J, L, K, M, N):
        return L * (K - np.exp((x - N) / J)) - M

    # Fit the data to the Hill function
    initial_guess = [ 10, 1, min(prices), 10, 1]  # Adjusted initial guesses
    params, _ = curve_fit(hill_func, days, prices, p0=initial_guess, maxfev=10000)
    J, K, L, M, N = params

    # Plot the data and the fit
    plt.figure(figsize=(12, 6))
    plt.scatter(days, prices, label="Actual Prices", color='blue', alpha=0.5)
    plt.plot(days, hill_func(days, *params), label=f"Fitted Curve:\n J={J:.2f}, K={K:.2f}, L={L:.2f}, M={M:.2f}, N={N:.2f}", color='red')
    plt.title("Bitcoin Price and Fitted Hill Curve")
    plt.xlabel("Days Since Start Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.show()

    # Return fitted parameters
    return J, K, L, M, N

# Call the function
params = fetch_and_fit_btc_prices()
print(f"Fitted Parameters: J={params[1]:.2f}, K={params[2]:.2f}, L={params[3]:.2f}, M={params[4]:.2f}, N={params[5]:.2f}")
