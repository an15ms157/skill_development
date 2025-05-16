import yfinance as yf
import matplotlib.pyplot as plt

def calculate_and_plot_fair_values(stock_symbol):
    # Fetch stock data using yfinance
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="10y")
    price = hist['Close']  # Daily closing price

    try:
        # Get financial info
        info = stock.info
        financials = stock.quarterly_financials.T
        if financials.empty:
            raise ValueError("No financial data available for this stock.")
        
        # Extract required data
        eps = info.get('trailingEps', None)  # Trailing 12-month EPS
        pe_ratio = info.get('forwardPE', None)  # Forward P/E ratio
        growth_rate = info.get('earningsGrowth', None)  # Growth rate in decimal (e.g., 0.1 for 10%)
        shares_outstanding = info.get('sharesOutstanding', 1)
        
        net_income = financials['Net Income'].dropna() if 'Net Income' in financials else None

        # 1. Price-to-Earnings (P/E) Fair Value
        fair_value_pe = eps * pe_ratio if eps and pe_ratio else None

        # 2. Dividend Discount Model (DDM)
        dividend = info.get('dividendRate', 0)  # Annual dividend per share
        assumed_growth_rate = 0.25  # Assume 4% growth rate if none is provided
        required_return = 0.08  # Assume 8% required return
        fair_value_ddm = dividend / (required_return - assumed_growth_rate) if dividend else None

        # 3. PEG Ratio Fair Value
        peg_ratio = info.get('pegRatio', None)
        fair_value_peg = eps * (pe_ratio / assumed_growth_rate) if eps and pe_ratio else None

        # 4. Net Asset Value (NAV)
        total_assets = info.get('totalAssets', None)
        total_liabilities = info.get('totalLiab', None)
        fair_value_nav = (total_assets - total_liabilities) / shares_outstanding if total_assets and total_liabilities else None

        # 5. Discounted Cash Flow (DCF) (simplified)
        next_year_cash_flow = info.get('operatingCashflow', None) / shares_outstanding if info.get('operatingCashflow', None) else None
        fair_value_dcf = next_year_cash_flow / (required_return - assumed_growth_rate) if next_year_cash_flow else None

        # 6. Peter Lynch Fair Value
        lynch_fair_value = eps * (growth_rate * 100) if eps and growth_rate else None

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(price.index, price, label='Stock Price', color='blue', linewidth=2)

        # Add horizontal lines for fair values
        if fair_value_pe:
            plt.axhline(y=fair_value_pe, color='red', linestyle='--', label='Fair Value (P/E)')
        if fair_value_ddm:
            plt.axhline(y=fair_value_ddm, color='green', linestyle='--', label='Fair Value (DDM)')
        if fair_value_peg:
            plt.axhline(y=fair_value_peg, color='orange', linestyle='--', label='Fair Value (PEG)')
        if fair_value_nav:
            plt.axhline(y=fair_value_nav, color='purple', linestyle='--', label='Fair Value (NAV)')
        if fair_value_dcf:
            plt.axhline(y=fair_value_dcf, color='brown', linestyle='--', label='Fair Value (DCF)')
        if lynch_fair_value:
            plt.axhline(y=lynch_fair_value, color='cyan', linestyle='--', label="Fair Value (Peter Lynch)")

        # Formatting the plot
        plt.ylim(min(price.min(), 0), max(price.max() * 1.5, fair_value_pe or 0, lynch_fair_value or 0))  # Adjust scale
        plt.title(f"{stock_symbol} Stock Price and Fair Values")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error calculating or plotting fair values: {e}")

# Example usage
calculate_and_plot_fair_values("MPWR")

