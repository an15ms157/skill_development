# This script downloads historical index data from Stooq (via pandas_datareader) and saves it as a JSON file with dates.
import pandas_datareader.data as web
import pandas as pd
import json
from datetime import datetime
import os

symbol = '^SPX'  # S&P 500 symbol for Stooq
start_date = '1900-01-01'
end_date = '2025-01-01'
#end_date = datetime.today().strftime('%Y-%m-%d')

try:
    data = web.DataReader(symbol, 'stooq', start=start_date, end=end_date)
    if not data.empty:
        data = data.sort_index()  # Stooq returns data in reverse order
        data = data.reset_index()
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        records = data.to_dict(orient='records')
        os.makedirs('./data', exist_ok=True)
        # Save using ticker name (remove special chars for filename)
        ticker_name = symbol.replace('^', '').replace('/', '_')
        json_path = f'./data/{ticker_name}_data.json'
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} records to {json_path}")
    else:
        print("No data downloaded.")
except Exception as e:
    print(f"Failed to download data: {e}")