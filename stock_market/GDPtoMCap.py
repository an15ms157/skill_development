import yfinance as yf
countries = ["USA", "CHN", "IND", "JPN", "DEU", "GBR", "FRA", "BRA", "ITA", "CAN", 
             "KOR", "RUS", "AUS", "ESP", "MEX", "IDN", "NLD", "TUR", "CHE", "SWE"]

	
def get_market_cap(country):
    indices_ticker = {
        "USA": "APPL",
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
    


    if country in indices_ticker:
        ticker_symbol = indices_ticker[country]
        try:
            index = yf.Ticker(ticker_symbol)
            print(index.info)
            market_cap = index.info.get("marketCap")
            if market_cap:
                return market_cap
            else:
                return "Market cap data not available."
        except Exception as e:
            return str(e)
    else:
        return "No main index ticker found for the given country."
   

# Example usage:
for country in countries:
    print(f"Market cap of {country}: {get_market_cap(country)}")

