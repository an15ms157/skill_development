import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

# Define the list of countries
countries = ["USA", "CHN", "IND", "JPN", "DEU", "GBR", "FRA", "BRA", "ITA", "CAN", 
             "KOR", "RUS", "AUS", "ESP", "MEX", "IDN", "NLD", "TUR", "CHE", "SWE"]

indices_ticker = {
        "USA": "^GSPC",
        "CHN": "000001.SS",
        "IND": "^BSESN",
        "JPN": "^N225",
        "DEU": "^GDAXI",
        "GBR": "^FTSE",
        "FRA": "^FCHI",
        "BRA": "^BVSP",
        "ITA": "FTSEMIB.MI",
        "CAN": "^GSPTSE",
        "KOR": "^KS11",
        "RUS": "IMOEX.ME",
        "AUS": "^AXJO",
        "ESP": "^IBEX",
        "MEX": "^MXX",
        "IDN": "^JKSE",
        "NLD": "^AEX",
        "TUR": "^XU100",
        "CHE": "^SSMI",
        "SWE": "^OMXS30"
    }


# Fetch GDP per capita data for the specified countries from 1950 to 2023
gdp_per_capita_data = wb.data.DataFrame('NY.GDP.PCAP.CD', countries, range(1950, 2024), index='time')


market_cap_data = {}
for country in countries[:5]:  # Fetching data for the first 5 countries
    ticker = indices_ticker[country]  # Yahoo Finance ticker format
    stock = yf.Ticker(ticker)
    market_cap_data[country] = stock.history(period="max").iloc[-1]['marketCap']

print(market_cap_data)

# Plot the data
plt.figure(figsize=(12, 8))  # Set the figure size
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 
          'lime', 'olive', 'teal', 'navy', 'gold', 'sienna', 'crimson', 'skyblue', 'lightgreen', 'salmon']

for i, country in enumerate(countries[:5]):
    plt.plot(gdp_per_capita_data.index, gdp_per_capita_data[country], label=country, color=colors[i])

plt.title('GDP Per Capita of Selected Countries (1950-2023)')
plt.xlabel('Year')
plt.ylabel('GDP Per Capita (Current US$)')
plt.yscale('log')  # Use a logarithmic scale for the y-axis
plt.xticks(np.arange(1950, 2024, 5))  # Set x-axis ticks to multiples of 5
plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1, 1))
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


