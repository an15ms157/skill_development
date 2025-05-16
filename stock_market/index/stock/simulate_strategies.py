import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def simulate_strategies(ticker, start_date, end_date):
    daily = yf.download(ticker, start=start_date, end=end_date, progress=False)
    daily['SMA200'] = daily['Close'].rolling(window=200).mean()

    monthly_price = daily['Close'].resample('M').last()
    monthly_sma   = daily['SMA200'].resample('M').last()

    data = pd.concat([monthly_price, monthly_sma], axis=1)
    data.columns = ['Price', 'SMA200']
    data = data.dropna()

    s1_units = s2_units = 0.0
    s1_invest = s2_invest = s2_savings = 0.0
    s1_vals = []
    s2_vals = []

    for price, sma in zip(data['Price'], data['SMA200']):
        # Strategy 1: invest $1/month
        s1_units   += 1.0 / price
        s1_invest  += 1.0

        # Strategy 2
        if price > sma:
            s2_savings += 1.0
        elif s2_savings >= 1.0:
            invest_amt   = s2_savings * 0.5
            s2_units    += invest_amt / price
            s2_invest   += invest_amt
            s2_savings  -= invest_amt
        else:
            s2_units    += 1.0 / price
            s2_invest   += 1.0

        s1_vals.append(s1_units * price)
        s2_vals.append(s2_units * price + s2_savings)

    final_price = data['Price'].iloc[-1]
    s1_value    = s1_units * final_price
    s2_value    = s2_units * final_price + s2_savings

    s1_roic = (s1_value - s1_invest) / s1_invest * 100
    s2_roic = (s2_value - s2_invest) / s2_invest * 100

    # Plot strategies
    plt.figure(figsize=(12,8))

    plt.subplot(2,1,1)
    plt.plot(data.index, data['Price'], label="Stock Price", color='gray')
    plt.title(f"{ticker} Stock Price")
    plt.ylabel("Price ($)")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(data.index, s1_vals, label=f"Strategy 1 (ROIC {s1_roic:.2f}%)", color='blue')
    plt.plot(data.index, s2_vals, label=f"Strategy 2 (ROIC {s2_roic:.2f}%)", color='green')
    plt.title("Investment Strategy Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Strategy 1 — Invested: ${s1_invest:.2f}, Value: ${s1_value:.2f}, ROIC: {s1_roic:.2f}%")
    print(f"Strategy 2 — Invested: ${s2_invest:.2f}, Value: ${s2_value:.2f}, ROIC: {s2_roic:.2f}%")

if __name__ == "__main__":
    simulate_strategies("AAPL", "2005-01-01", "2025-01-01")

