import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy_config import WEIGHTS_BY_YEAR, sip_config

def get_weights_by_year(date):
    for year, weights in WEIGHTS_BY_YEAR.items():
        if date.year < year:
            return weights

def calculate_sip_returns(input_folder):
    start_date = pd.to_datetime(sip_config["start_date"])
    end_date = pd.to_datetime(sip_config["end_date"])
    monthly_investment = sip_config["monthly_investment"]

    # Read index data
    index_data = {}
    for country in get_weights_by_year(start_date).keys():
        file_path = os.path.join(input_folder, f"{country}_usd_index_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%d', errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            df.set_index('Date', inplace=True)

            # Ensure only numeric columns are converted to float
            try:
                df.iloc[:, 0] = df.iloc[:, 0].replace('[\$,]', '', regex=True).astype(float)
                index_data[country] = df.iloc[:, 0]
                print(f"[INFO] Successfully loaded data for {country}.")
            except ValueError as e:
                print(f"[ERROR] Failed to process {file_path}: {e}")
                continue
        else:
            print(f"[WARNING] File not found: {file_path}")

    if not index_data:
        print("[ERROR] No valid index data loaded. Exiting.")
        return pd.DataFrame(columns=['Total_Value', 'Total_Investment', 'ROIC', 'Savings'])  # Return an empty DataFrame with expected columns

    all_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    portfolio = pd.DataFrame(index=all_dates)
    total_investment = 0
    units_held = {country: 0 for country in index_data.keys()}
    savings = 0
    savings_history = []

    for i, date in enumerate(all_dates):
        weights = get_weights_by_year(date)

        # Initialize additional_investment to avoid UnboundLocalError
        additional_investment = 0

        for country, weight in weights.items():
            if country not in index_data:
                print(f"[WARNING] No data for {country}. Skipping.")
                continue

            investment_amount = monthly_investment * weight
            total_investment += investment_amount

            index_series = index_data[country]
            future_dates = index_series.index[index_series.index >= date]
            if future_dates.empty:
                print(f"[WARNING] No future data for {country} on {date}. Skipping.")
                continue
            closest_date = future_dates[0]
            price = index_series[closest_date]

            # Defensive conversion to float
            try:
                price = float(str(price).replace(',', '').strip())
            except ValueError:
                print(f"[ERROR] Invalid price '{price}' for {country} on {closest_date}")
                continue

            new_units = investment_amount / price
            units_held[country] += new_units
            portfolio.loc[date, f'{country}_Value'] = units_held[country] * price

        # Debug: Check if Total_Value is being updated
        portfolio.loc[date, 'Total_Value'] = portfolio.filter(like='_Value').loc[date].sum()
        if portfolio.loc[date, 'Total_Value'] == 0:
            print(f"[DEBUG] Total Value is 0 on {date}. Check data and logic.")

        current_value = portfolio['Total_Value'].iloc[i] if i < len(portfolio) else 0

        if i >= 12:
            savings += monthly_investment

            sma_50 = portfolio['Total_Value'].iloc[:i + 1].rolling(window=50).mean().iloc[-1] if i >= 49 else np.nan
            sma_100 = portfolio['Total_Value'].iloc[:i + 1].rolling(window=100).mean().iloc[-1] if i >= 99 else np.nan
            sma_20 = portfolio['Total_Value'].iloc[:i + 1].rolling(window=20).mean().iloc[-1] if i >= 19 else np.nan

            if savings >= 1:
                if not np.isnan(sma_100) and current_value < sma_100:
                    additional_investment = 0.5 * savings
                elif not np.isnan(sma_50) and current_value < sma_50:
                    additional_investment = 0.3 * savings
                elif not np.isnan(sma_20) and current_value < sma_20:
                    additional_investment = 0.2 * savings

            savings -= additional_investment
            total_investment += additional_investment

            for country in units_held.keys():
                if country not in index_data:
                    continue
                index_series = index_data[country]
                future_dates = index_series.index[index_series.index >= date]
                if future_dates.empty:
                    continue
                closest_date = future_dates[0]
                price = index_series[closest_date]

                # Defensive conversion to float again
                try:
                    price = float(str(price).replace(',', '').strip())
                except ValueError:
                    print(f"[ERROR] Invalid price '{price}' for {country} on {closest_date}")
                    continue

                new_units = additional_investment / price
                units_held[country] += new_units
                portfolio.loc[date, f'{country}_Value'] = units_held[country] * price

        savings_history.append(savings)

        # Debug info
        print(f"Date: {date.strftime('%Y-%m-%d')}")
        print(f"Total Value: {current_value:.2f}")
        print(f"Savings: {savings:.2f}")
        print(f"Additional Investment: {additional_investment:.2f}")
        print("-" * 40)

    portfolio['Total_Value'] = portfolio.filter(like='_Value').sum(axis=1)
    portfolio['Total_Investment'] = np.cumsum([monthly_investment] * len(portfolio))
    portfolio['ROIC'] = portfolio['Total_Value'] / portfolio['Total_Investment'] - 1
    portfolio['Savings'] = savings_history

    return portfolio

def plot_sip_performance(portfolio, output_folder):
    if portfolio is None or portfolio.empty:
        print("[ERROR] Portfolio is empty or None. Exiting plot generation.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    portfolio['SMA_50'] = portfolio['Total_Value'].rolling(window=50).mean()
    portfolio['SMA_100'] = portfolio['Total_Value'].rolling(window=100).mean()
    portfolio['SMA_20'] = portfolio['Total_Value'].rolling(window=20).mean()

    ax1.plot(portfolio.index, portfolio['Total_Value'], label='Portfolio Value', linewidth=2)
    ax1.plot(portfolio.index, portfolio['Total_Investment'], label='Net Investment', linestyle='--', color='purple')
    ax1.plot(portfolio.index, portfolio['SMA_50'], label='50-day SMA', linestyle='--', color='orange')
    ax1.plot(portfolio.index, portfolio['SMA_100'], label='100-day SMA', linestyle='--', color='red')
    ax1.plot(portfolio.index, portfolio['SMA_20'], label='20-day SMA', linestyle='--', color='blue')
    ax1.set_title('SIP Strategy: Portfolio Value, SMAs, and Net Investment')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('USD')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(portfolio.index, portfolio['ROIC'], label='ROIC', color='green')
    ax2.set_ylabel('ROIC (x)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.grid(True, alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(portfolio.index, portfolio['Savings'], label='Savings', linestyle='--', color='blue')
    ax3.set_ylabel('Savings (USD)', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')

    ax2.set_title('SIP Strategy: ROIC and Savings (Different Scales)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plot_file = os.path.join(output_folder, "sip_strategy_performance.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()

    print(f"Final Portfolio Value: ${portfolio['Total_Value'].iloc[-1]:.2f}")
    print(f"Total Investment: ${portfolio['Total_Investment'].iloc[-1]:.2f}")
    print(f"Overall ROIC: {portfolio['ROIC'].iloc[-1]:.2f}")
    print(f"Saved plot to: {plot_file}")

if __name__ == "__main__":
    input_folder = "usd_index_data"
    output_folder = "plots"

    portfolio = calculate_sip_returns(input_folder)
    plot_sip_performance(portfolio, output_folder)
