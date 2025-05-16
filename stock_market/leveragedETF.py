import yfinance as yf
import matplotlib.pyplot as plt

# Define start and end dates and leverage factor
START = "2000-01-01"
END = "2022-12-31"
leverage = 3
maintanence=0.00001918 
#maintanence=0
taxcut=0.73
# Fetch historical data for the S&P 500 index
#data = yf.download("^GSPC", start=START, end=END)
data = yf.download("NVDA", start=START, end=END)
# Extract the adjusted close price as the base index
prices = data['Adj Close']

# Construct a new index that moves daily by a factor of 3 of the base index's price changes
new_index = prices.copy()
for i in range(1, len(prices)):
    new_index.iloc[i] = new_index.iloc[i-1] + leverage * (prices.iloc[i] - prices.iloc[i-1])
    new_index.iloc[i] = new_index.iloc[i]-maintanence*abs(new_index.iloc[i]) 

# Calculate gains for successive 10-year periods
start_year = int(START[:4])
end_year = int(END[:4])
years = []
sp500_gains = []
leveraged_gains = []
leveraged_gainsPLUStax	= []

N=1
for year in range(start_year, end_year, N):
    # Define the 10-year period
    period_start = f"{year}-01-01"
    period_end = f"{min(year+N, end_year)}-12-31"
    print(period_start, period_end)
    # Extract data for the period
    sp500_period = prices[period_start:period_end]
    new_index_period = new_index[period_start:period_end]
    
    # Calculate percentage gains for the period
    sp500_gain = (sp500_period.iloc[-1] - sp500_period.iloc[0]) / sp500_period.iloc[0] * 100
    new_index_gain = (new_index_period.iloc[-1] - new_index_period.iloc[0]) / new_index_period.iloc[0] * 100
    
    # Store results
    years.append(f"{year}-{year + N}")
    sp500_gains.append(sp500_gain)
    leveraged_gains.append(new_index_gain)
    leveraged_gainsPLUStax.append(taxcut*new_index_gain)


# Compute 50-day and 200-day moving averages for the leveraged index
ma_50 = new_index.rolling(window=50).mean()
ma_200 = new_index.rolling(window=200).mean()
positive_sum = sum(num for num in leveraged_gainsPLUStax if num > 0)

print(sum(sp500_gains), sum(leveraged_gains), sum(leveraged_gainsPLUStax), positive_sum)

# Plot the original S&P 500 index and the constructed index
plt.figure(figsize=(14, 7))
plt.plot(prices, label="S&P 500 Index (Base)", color="blue")
plt.plot(new_index, label="Constructed Index", color="orange")
plt.plot(ma_50, label="50-Day MA Lev", color="green", linestyle="--")
plt.plot(ma_200, label="200-Day MA Lev", color="red", linestyle="--")
plt.title("S&P 500 vs Constructed Index")
plt.xlabel("Date")
plt.ylabel("Index Value")
#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# Plot the gains in a single plot
plt.figure(figsize=(14, 7))
plt.plot(years, sp500_gains, label="S&P 500 Gain", marker='o', color="blue")
plt.plot(years, leveraged_gains, label="Leveraged Index Gain", marker='o', color="green")
plt.plot(years, leveraged_gainsPLUStax, label="Leveraged Index Gain - TAX", marker='o', color="red")
plt.title("Comparison of Gains Over Successive 10-Year Periods")
plt.xlabel(f"{N} Year Period")
plt.ylabel("Percentage Gain")
plt.xticks(rotation=45)
#plt.yscale('log')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()

