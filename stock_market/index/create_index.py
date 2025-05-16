import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

from strategy_config import WEIGHTS_BY_YEAR, sip_config, INDEX_PATHS

OUTPUT_DIR = 'index_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_weight_config_for_year(year):
    for key in WEIGHTS_BY_YEAR:
        if key == float('inf') or (isinstance(key, str) and year < int(key.split()[1])):
            return WEIGHTS_BY_YEAR[key]
    return WEIGHTS_BY_YEAR[float('inf')]

def load_data():
    index_data = {}
    for country, path in INDEX_PATHS.items():
        try:
            # Skip the first 3 rows (non-data), and set actual headers
            df = pd.read_csv(path, skiprows=3, header=None, names=["Date", "Price"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.dropna(subset=["Date", "Price"], inplace=True)
            df.set_index("Date", inplace=True)
            index_data[country] = df["Price"].astype(float).sort_index()
            print(f"[INFO] Loaded {country}: {len(index_data[country])} records")
        except Exception as e:
            print(f"[ERROR] Failed to load {country}: {e}")
    return index_data

def calculate_individual_investment(index_data, sip_config):
    start_date = pd.to_datetime(sip_config['start_date'])
    end_date = pd.to_datetime(sip_config['end_date'])
    monthly_investment = sip_config['monthly_investment']

    investment_values = {country: pd.Series(dtype=float) for country in index_data}
    units_held = {country: 0.0 for country in index_data}

    all_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

    for date in all_dates:
        year = date.year
        weight_config = get_weight_config_for_year(year)

        for country, series in index_data.items():
            if country not in weight_config or weight_config[country] <= 0:
                continue

            # Pick the closest forward date with data
            if date not in series.index:
                try:
                    pos = series.index.searchsorted(date, side='left')
                    if pos >= len(series):
                        continue  # No forward date available
                    nearest = series.index[pos]
                except KeyError:
                    continue
            else:
                nearest = date

            price = series.get(nearest)
            if pd.isna(price) or price == 0:
                continue

            units = (monthly_investment * weight_config[country]) / price
            units_held[country] += units

        # Update investment value for the current month
        for country, series in index_data.items():
            valid_dates = series[series.index >= start_date].index
            investment_values[country] = units_held[country] * series

    return investment_values

def calculate_global_index(investment_values):
    combined = pd.DataFrame(investment_values)
    if combined.empty:
        print("[ERROR] No data available to calculate Global Index.")
        return combined
    combined['Global Index'] = combined.sum(axis=1)
    combined['Global Index'] /= combined['Global Index'].iloc[0]  # Normalize
    return combined

def calculate_monthly_investment_for_global(index_data, sip_config):
    start_date = pd.to_datetime(sip_config['start_date'])
    end_date = pd.to_datetime(sip_config['end_date'])
    
    # Global index investment (1 dollar each month)
    global_investment = pd.Series(0.0, index=pd.date_range(start=start_date, end=end_date, freq='MS'))
    
    # Calculate the global index value based on weighted investments
    for date in global_investment.index:
        weight_config = get_weight_config_for_year(date.year)
        
        for country, series in index_data.items():
            if country not in weight_config or weight_config[country] <= 0:
                continue

            # Pick the closest forward date with data
            if date not in series.index:
                try:
                    pos = series.index.searchsorted(date, side='left')
                    if pos >= len(series):
                        continue  # No forward date available
                    nearest = series.index[pos]
                except KeyError:
                    continue
            else:
                nearest = date

            price = series.get(nearest)
            if pd.isna(price) or price == 0:
                continue

            # Investment in the country based on its weight
            global_investment[date] += 1 * weight_config[country] / price

    return global_investment

def plot_global_index(global_df):
    plt.figure(figsize=(12, 6))
    global_df['Global Index'].plot()
    plt.title('Global Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    path = os.path.join(OUTPUT_DIR, 'global_index_plot.png')
    plt.savefig(path)
    print(f"Index plot saved to: {path}")
    plt.close()

def plot_individual_investments(investment_values, global_investment):
    plt.figure(figsize=(12, 6))
    start_date = pd.to_datetime(sip_config['start_date'])  # Ensure start date aligns with strategy config

    # Plot individual country investments
    for country, series in investment_values.items():
        if not series.empty:
            filtered_series = series[series.index >= start_date]  # Filter data starting from the strategy's start date
            if not filtered_series.empty:
                (filtered_series / filtered_series.dropna().iloc[0]).plot(label=country)

    # Plot global index investment
    global_investment_filtered = global_investment[global_investment.index >= start_date]
    if not global_investment_filtered.empty:
        (global_investment_filtered / global_investment_filtered.iloc[0]).plot(label="Global Index", color="black", linewidth=2)

    plt.yscale('log')  # Set y-axis to log scale
    plt.title('Investment Value by Country and Global Index (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (start at $1)')
    plt.legend()
    plt.grid(True)
    path = os.path.join(OUTPUT_DIR, 'individual_investment_plot_with_global_log.png')
    plt.savefig(path)
    print(f"Individual investment plot with global index (log scale) saved to: {path}")
    plt.close()

def plot_combined_investment(index_data, investment_values, global_investment):
    plt.figure(figsize=(12, 6))

    # Plot individual country investments
    for country, series in investment_values.items():
        if not series.empty:
            (series / series.dropna().iloc[0]).plot(label=f"{country} Investment")

    # Plot global index investment
    global_investment_normalized = global_investment / global_investment.iloc[0]
    global_investment_normalized.plot(label="Global Index Investment", color="black", linewidth=2)

    plt.title('Investment Value by Country and Global Index (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (start at $1)')
    plt.legend()
    plt.grid(True)
    path = os.path.join(OUTPUT_DIR, 'combined_investment_plot.png')
    plt.savefig(path)
    print(f"Combined investment plot saved to: {path}")
    plt.close()

def main():
    index_data = load_data()
    investment_values = calculate_individual_investment(index_data, sip_config)
    global_df = calculate_global_index(investment_values)
    
    # Calculate global investment (1 dollar per month into the global index)
    global_investment = calculate_monthly_investment_for_global(index_data, sip_config)

    if not global_df.empty:
        global_df.to_csv(os.path.join(OUTPUT_DIR, 'global_index.csv'))
        print(f"Index data saved to: {os.path.join(OUTPUT_DIR, 'global_index.csv')}")
        plot_global_index(global_df)

    plot_individual_investments(investment_values, global_investment)

    # Plot the combined investment chart with the global and individual country investments
    plot_combined_investment(index_data, investment_values, global_investment)

    print("[DEBUG] First date and value of investment for each index:")
    for country, series in investment_values.items():
        if not series.empty:
            first_date = series.dropna().index[0]
            first_value = series.dropna().iloc[0]
            print(f"{country}: Date = {first_date.date()}, Value = {first_value:.2f}")
        else:
            print(f"{country}: No data available")

if __name__ == '__main__':
    main()
