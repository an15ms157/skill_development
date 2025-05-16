import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import os

from dcf_monte_carlo import DCFModel, run_monte_carlo, extract_dcf_inputs

def calculate_historical_dcf(ticker, data_dir="data", start_year=1998, end_year=2025):
    # Load data to determine the earliest available year
    income_file = f"{data_dir}/{ticker}/{ticker} Income Statement (Annual) - Discounting Cash Flows.xlsx"
    income_df = pd.read_excel(income_file, sheet_name=0, header=0, index_col=0)
    income_df.columns = income_df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
    income_df.columns = [col if col == 'LTM' else col.split(' ')[0].split('-')[0] for col in income_df.columns]

    # Find all years in the columns (assuming years are numeric)
    all_years = [col for col in income_df.columns if col.isdigit()]
    all_years = [int(year) for year in all_years]
    if not all_years:
        print(f"No valid years found in data for {ticker}. Exiting.")
        return ([], [], [], [], [], [], [], 'USD', [], [], [], [])

    # Determine the earliest available year
    earliest_year = min(all_years)
    print(f"Earliest available year for {ticker}: {earliest_year}")

    # Adjust start_year if it's before the earliest available year
    if start_year < earliest_year:
        print(f"Requested start_year {start_year} is before earliest available year {earliest_year}. Adjusting start_year to {earliest_year}.")
        start_year = earliest_year

    years = []
    # Simple model results
    simple_mean_fair_values = []
    simple_sigma2_low = []
    simple_sigma2_high = []
    # Enhanced model results
    enhanced_mean_fair_values = []
    enhanced_sigma2_low = []
    enhanced_sigma2_high = []
    # LTM results for the current date
    ltm_simple_mean = None
    ltm_simple_sigma2_low = None
    ltm_simple_sigma2_high = None
    ltm_enhanced_mean = None
    ltm_enhanced_sigma2_low = None
    ltm_enhanced_sigma2_high = None

    # Check if LTM data is available
    if 'LTM' in income_df.columns:
        print(f"\nProcessing LTM data for {end_year} for {ticker}...")
        try:
            inputs = extract_dcf_inputs(ticker, data_dir=data_dir, start_year=end_year, end_year=end_year)
            if inputs is None:
                print("Skipping LTM: Unable to extract inputs.")
            elif pd.isna(inputs['starting_fcf']) or pd.isna(inputs['shares_outstanding']):
                print("Skipping LTM: Missing critical inputs.")
            else:
                # Simple Model for LTM
                print("Running Simple Model for LTM...")
                simple_model = DCFModel(
                    starting_fcf=inputs['starting_fcf'],
                    wacc=inputs['wacc'],
                    high_growth_rate=inputs['high_growth_rate'],
                    terminal_growth_rate=inputs['terminal_growth_rate'],
                    net_debt=inputs['net_debt'],
                    shares_outstanding=inputs['shares_outstanding'],
                    use_enhanced_model=False,
                    roic=inputs['roic'],
                    ticker=ticker,
                    reinvestment_cap=inputs['reinvestment_cap']
                )

                simple_results = run_monte_carlo(
                    simple_model,
                    num_simulations=10000,
                    wacc_range=(0.06, 0.12),
                    growth_range=(0.05, 0.15),
                    terminal_growth_range=(0.02, 0.04),
                    fcf_error=0.1
                )

                if len(simple_results) > 0:
                    ltm_simple_mean = np.mean(simple_results)
                    ltm_simple_std_dev = np.std(simple_results)
                    ltm_simple_sigma2_low = max(ltm_simple_mean - 2 * ltm_simple_std_dev, 0)
                    ltm_simple_sigma2_high = ltm_simple_mean + 2 * ltm_simple_std_dev
                    print(f"LTM Simple Model Mean Fair Value: {ltm_simple_mean:.2f}")
                    print(f"LTM Simple Model 2σ Range: {ltm_simple_sigma2_low:.2f} to {ltm_simple_sigma2_high:.2f}")

                # Enhanced Model for LTM
                print("Running Enhanced Model for LTM...")
                enhanced_model = DCFModel(
                    starting_fcf=inputs['starting_fcf'],
                    wacc=inputs['wacc'],
                    high_growth_rate=inputs['high_growth_rate'],
                    terminal_growth_rate=inputs['terminal_growth_rate'],
                    net_debt=inputs['net_debt'],
                    shares_outstanding=inputs['shares_outstanding'],
                    use_enhanced_model=True,
                    roic=inputs['roic'],
                    ticker=ticker,
                    reinvestment_cap=inputs['reinvestment_cap']
                )

                enhanced_results = run_monte_carlo(
                    enhanced_model,
                    num_simulations=10000,
                    wacc_range=(0.06, 0.12),
                    growth_range=(0.05, 0.15),
                    terminal_growth_range=(0.02, 0.04),
                    fcf_error=0.1
                )

                if len(enhanced_results) > 0:
                    ltm_enhanced_mean = np.mean(enhanced_results)
                    ltm_enhanced_std_dev = np.std(enhanced_results)
                    ltm_enhanced_sigma2_low = max(ltm_enhanced_mean - 2 * ltm_enhanced_std_dev, 0)
                    ltm_enhanced_sigma2_high = ltm_enhanced_mean + 2 * ltm_enhanced_std_dev
                    print(f"LTM Enhanced Model Mean Fair Value: {ltm_enhanced_mean:.2f}")
                    print(f"LTM Enhanced Model 2σ Range: {ltm_enhanced_sigma2_low:.2f} to {ltm_enhanced_sigma2_high:.2f}")
        except Exception as e:
            print(f"Error processing LTM data: {e}")

    # Annual DCF calculations for both models
    for year in range(end_year, start_year - 1, -1):
        print(f"\nProcessing year {year} for {ticker}...")

        try:
            inputs = extract_dcf_inputs(ticker, data_dir=data_dir, start_year=year, end_year=year)
            if inputs is None:
                print(f"Skipping year {year}: Unable to extract inputs.")
                continue

            if pd.isna(inputs['starting_fcf']) or pd.isna(inputs['shares_outstanding']):
                print(f"Skipping year {year}: Missing critical inputs.")
                continue

            # Simple Model
            print(f"Running Simple Model for year {year}...")
            simple_model = DCFModel(
                starting_fcf=inputs['starting_fcf'],
                wacc=inputs['wacc'],
                high_growth_rate=inputs['high_growth_rate'],
                terminal_growth_rate=inputs['terminal_growth_rate'],
                net_debt=inputs['net_debt'],
                shares_outstanding=inputs['shares_outstanding'],
                use_enhanced_model=False,
                roic=inputs['roic'],
                ticker=ticker,
                reinvestment_cap=inputs['reinvestment_cap']
            )

            simple_results = run_monte_carlo(
                simple_model,
                num_simulations=10000,
                wacc_range=(0.06, 0.12),
                growth_range=(0.05, 0.15),
                terminal_growth_range=(0.02, 0.04),
                fcf_error=0.1
            )

            if len(simple_results) == 0:
                print(f"No valid simple model simulation results for year {year}. Skipping.")
                continue

            # Enhanced Model
            print(f"Running Enhanced Model for year {year}...")
            enhanced_model = DCFModel(
                starting_fcf=inputs['starting_fcf'],
                wacc=inputs['wacc'],
                high_growth_rate=inputs['high_growth_rate'],
                terminal_growth_rate=inputs['terminal_growth_rate'],
                net_debt=inputs['net_debt'],
                shares_outstanding=inputs['shares_outstanding'],
                use_enhanced_model=True,
                roic=inputs['roic'],
                ticker=ticker,
                reinvestment_cap=inputs['reinvestment_cap']
            )

            enhanced_results = run_monte_carlo(
                enhanced_model,
                num_simulations=10000,
                wacc_range=(0.06, 0.12),
                growth_range=(0.05, 0.15),
                terminal_growth_range=(0.02, 0.04),
                fcf_error=0.1
            )

            if len(enhanced_results) == 0:
                print(f"No valid enhanced model simulation results for year {year}. Skipping.")
                continue

            # Calculate statistics for Simple Model
            simple_mean_val = np.mean(simple_results)
            simple_std_dev = np.std(simple_results)

            # Calculate statistics for Enhanced Model
            enhanced_mean_val = np.mean(enhanced_results)
            enhanced_std_dev = np.std(enhanced_results)

            # Store results
            years.append(year)
            simple_mean_fair_values.append(simple_mean_val)
            simple_sigma2_low.append(max(simple_mean_val - 2 * simple_std_dev, 0))
            simple_sigma2_high.append(simple_mean_val + 2 * simple_std_dev)

            enhanced_mean_fair_values.append(enhanced_mean_val)
            enhanced_sigma2_low.append(max(enhanced_mean_val - 2 * enhanced_std_dev, 0))
            enhanced_sigma2_high.append(enhanced_mean_val + 2 * enhanced_std_dev)

            print(f"Year {year} - Simple Model Mean Fair Value: {simple_mean_val:.2f}")
            print(f"Simple Model 2σ Range: {simple_sigma2_low[-1]:.2f} to {simple_sigma2_high[-1]:.2f}")
            print(f"Year {year} - Enhanced Model Mean Fair Value: {enhanced_mean_val:.2f}")
            print(f"Enhanced Model 2σ Range: {enhanced_sigma2_low[-1]:.2f} to {enhanced_sigma2_high[-1]:.2f}")

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue

    return (years, simple_mean_fair_values, simple_sigma2_low, simple_sigma2_high,
            enhanced_mean_fair_values, enhanced_sigma2_low, enhanced_sigma2_high,
            inputs['currency'] if 'inputs' in locals() else 'USD',
            ltm_simple_mean, ltm_simple_sigma2_low, ltm_simple_sigma2_high,
            ltm_enhanced_mean, ltm_enhanced_sigma2_low, ltm_enhanced_sigma2_high)

def fetch_weekly_prices(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1wk")
        
        if df.empty:
            print(f"No price data available for {ticker} from {start_date} to {end_date}")
            return pd.DataFrame(columns=['Date', 'Price'])
        
        print("Columns in yfinance DataFrame:", df.columns)
        print(f"Earliest price data: {df.index.min()}")
        print(f"Latest price data: {df.index.max()}")
        
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close' if 'Close' in df.columns else None
        if price_col is None:
            print("Error: No 'Adj Close' or 'Close' column found in price data")
            return pd.DataFrame(columns=['Date', 'Price'])
        
        df = df[[price_col]].reset_index()
        df.rename(columns={price_col: 'Price', 'Date': 'Date'}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        return df
    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame(columns=['Date', 'Price'])

def plot_dcf_and_prices(ticker, years, simple_mean_fair_values, simple_sigma2_low, simple_sigma2_high,
                        enhanced_mean_fair_values, enhanced_sigma2_low, enhanced_sigma2_high,
                        price_data, currency,
                        ltm_simple_mean, ltm_simple_sigma2_low, ltm_simple_sigma2_high,
                        ltm_enhanced_mean, ltm_enhanced_sigma2_low, ltm_enhanced_sigma2_high):
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))

    # Plot weekly stock prices
    if not price_data.empty:
        plt.plot(price_data['Date'], price_data['Price'], label='Weekly Price', color='black', linewidth=1)

    # Prepare dates for annual data (end of year: December 31st)
    year_dates = [pd.to_datetime(f"{year}-12-31") for year in years]

    # Plot Simple Model annual data
    plt.plot(year_dates, simple_mean_fair_values, label='Simple DCF Mean Fair Value', color='green', marker='o')
    plt.fill_between(year_dates, simple_sigma2_low, simple_sigma2_high, color='green', alpha=0.2, label='Simple 2σ Range')

    # Plot Enhanced Model annual data
    plt.plot(year_dates, enhanced_mean_fair_values, label='Enhanced DCF Mean Fair Value', color='blue', marker='o')
    plt.fill_between(year_dates, enhanced_sigma2_low, enhanced_sigma2_high, color='blue', alpha=0.2, label='Enhanced 2σ Range')

    # Plot LTM data at the current date (April 29, 2025)
    current_date = pd.to_datetime("2025-04-29")
    if ltm_simple_mean is not None:
        plt.scatter([current_date], [ltm_simple_mean], color='green', marker='*', s=100, label='Simple DCF LTM (2025)')
        plt.errorbar([current_date], [ltm_simple_mean], yerr=[[ltm_simple_mean - ltm_simple_sigma2_low], [ltm_simple_sigma2_high - ltm_simple_mean]], 
                     color='green', capsize=5, alpha=0.5)
    if ltm_enhanced_mean is not None:
        plt.scatter([current_date], [ltm_enhanced_mean], color='blue', marker='*', s=100, label='Enhanced DCF LTM (2025)')
        plt.errorbar([current_date], [ltm_enhanced_mean], yerr=[[ltm_enhanced_mean - ltm_enhanced_sigma2_low], [ltm_enhanced_sigma2_high - ltm_enhanced_mean]], 
                     color='blue', capsize=5, alpha=0.5)

    # Set y-axis limits based on price_data only
    if not price_data.empty:
        lowest_price = min(price_data['Price'])
        highest_price = max(price_data['Price'])
        plt.ylim(lowest_price * 0.1, highest_price * 10)
    else:
        print("Warning: No price data to set y-axis limits. Using default range based on DCF values.")
        all_values = (simple_mean_fair_values + simple_sigma2_low + simple_sigma2_high +
                      enhanced_mean_fair_values + enhanced_sigma2_low + enhanced_sigma2_high)
        if ltm_simple_mean is not None:
            all_values.extend([ltm_simple_mean, ltm_simple_sigma2_low, ltm_simple_sigma2_high])
        if ltm_enhanced_mean is not None:
            all_values.extend([ltm_enhanced_mean, ltm_enhanced_sigma2_low, ltm_enhanced_sigma2_high])
        if all_values:
            lowest_price = min(all_values)
            highest_price = max(all_values)
            plt.ylim(lowest_price * 0.1, highest_price * 3)
        else:
            plt.ylim(1, 10000)

    # Set x-axis ticks for each year
    all_years = years.copy()
    # Do not append 2025 to all_years since LTM is now at the current date, not year-end
    x_ticks = [pd.to_datetime(f"{year}-12-31") for year in all_years]
    x_labels = [str(year) for year in all_years]
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45)

    # Ensure the x-axis includes the full range, including the current date
    if all_years:
        plt.xlim(pd.to_datetime(f"{min(all_years)}-07-01"), pd.to_datetime("2025-06-01"))

    plt.title(f'Historical DCF Fair Value vs. Weekly Stock Price for {ticker}')
    plt.xlabel('Year')
    plt.ylabel(f'Price ({currency})')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{ticker}_dcf_historical.png")
    plt.show()

def convert_to_numeric(value, is_shares=False, is_price=False):
    if isinstance(value, pd.Series):
        value = value.item()
    if isinstance(value, str):
        if '%' in value:
            value = value.replace('%', '').strip()
            try:
                return float(value) / 100
            except ValueError:
                return np.nan
        value = value.replace(',', '').replace('₹', '').replace('INR', '').replace(' million', '').strip()
        try:
            scale = 1e6
            if is_shares or is_price:
                scale = 1
            return float(value) * scale
        except ValueError:
            return np.nan
    elif isinstance(value, (int, float)):
        scale = 1e6
        if is_shares or is_price:
            scale = 1
        return float(value) * scale
    return np.nan

def main():
    ticker = "SBIN.NS"
    data_dir = "data"
    start_year = 1980
    end_year = 2025

    # Set the end date to today to ensure we fetch the latest price data
    current_date = datetime(2025, 4, 29)

    (years, simple_mean_fair_values, simple_sigma2_low, simple_sigma2_high,
     enhanced_mean_fair_values, enhanced_sigma2_low, enhanced_sigma2_high,
     currency,
     ltm_simple_mean, ltm_simple_sigma2_low, ltm_simple_sigma2_high,
     ltm_enhanced_mean, ltm_enhanced_sigma2_low, ltm_enhanced_sigma2_high) = calculate_historical_dcf(
        ticker, data_dir=data_dir, start_year=start_year, end_year=end_year
    )

    if not years:
        print("No DCF data available to plot.")
        return

    start_date = datetime(years[-1], 1, 1)  # Use the earliest year from the adjusted years list
    end_date = current_date  # Fetch price data up to today
    price_data = fetch_weekly_prices(ticker, start_date, end_date)

    if price_data.empty:
        print("Warning: No price data available. Plotting only DCF values.")

    plot_dcf_and_prices(
        ticker, years, simple_mean_fair_values, simple_sigma2_low, simple_sigma2_high,
        enhanced_mean_fair_values, enhanced_sigma2_low, enhanced_sigma2_high,
        price_data, currency,
        ltm_simple_mean, ltm_simple_sigma2_low, ltm_simple_sigma2_high,
        ltm_enhanced_mean, ltm_enhanced_sigma2_low, ltm_enhanced_sigma2_high
    )

if __name__ == "__main__":
    main()