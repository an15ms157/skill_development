import pandas as pd
import numpy as np

def perform_sip_simulation(df, rolling_years=30, fixed_day=1):
    """
    Perform a 30-year rolling window SIP simulation by investing 1 USD on a fixed trading day each month.
    If the fixed day isn't a trading day, use the next available trading day in that month.
    
    Parameters:
    - df: DataFrame with S&P 500 data (columns: Date, Year, Month, Index)
    - rolling_years: Number of years for each rolling window (default: 30)
    - fixed_day: Day of the month to invest (default: 1, i.e., 1st trading day)
    
    Returns a list of dictionaries containing ROIC results for each window.
    """
    print(f"Starting 30-Year Fixed Day SIP Simulation (Investing on day {fixed_day} of each month)...")

    roic_results = []
    min_year = df['Year'].min()
    max_year = df['Year'].max() - rolling_years + 1

    for start_year in range(min_year, max_year + 1):
        end_year = start_year + rolling_years - 1
        df_period = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()

        df_period['YearMonth'] = df_period['Year'].astype(str) + '-' + df_period['Month'].astype(str).str.zfill(2)
        grouped = df_period.groupby('YearMonth')

        investment_dates = []
        investment_prices = []

        for _, group in grouped:
            if not group.empty:
                # Reset index for the group to work with local indices
                group = group.reset_index(drop=True)
                # Add day of the month for filtering
                group['Day'] = group['Date'].dt.day
                # Find the first trading day on or after the fixed day
                eligible_days = group[group['Day'] >= fixed_day]
                if not eligible_days.empty:
                    # Take the first available trading day on or after fixed_day
                    idx = eligible_days.index[0]
                    investment_dates.append(group.iloc[idx]['Date'])
                    investment_prices.append(group.iloc[idx]['Index'])
                else:
                    # If no days are available on or after fixed_day, take the last day of the month
                    idx = len(group) - 1
                    investment_dates.append(group.iloc[idx]['Date'])
                    investment_prices.append(group.iloc[idx]['Index'])

        if not investment_prices:
            continue

        units = 1 / np.array(investment_prices)
        total_units = np.sum(units)
        final_price = float(df_period['Index'].iloc[-1].item())
        final_value = total_units * final_price

        total_invested = len(investment_prices)
        net_return = (final_value - total_invested) / total_invested * 100  # Net ROIC %

        roic_results.append({
            'StartYear': start_year,
            'EndYear': end_year,
            'NetROIC': net_return,
            'StartDate': df_period['Date'].iloc[0],
            'EndDate': df_period['Date'].iloc[-1],
            'IndexValues': df_period[['Date', 'Index']]
        })

    # Save ROIC vs StartYear to CSV
    results_df = pd.DataFrame(roic_results)
    roic_df = results_df[['StartYear', 'NetROIC']].copy()
    roic_df.rename(columns={'StartYear': 'Start_Year', 'NetROIC': 'Net_ROIC_Percent'}, inplace=True)
    csv_filename = get_plot_filenames()['csv_filename']
    roic_df.to_csv(csv_filename, index=False)
    print(f"ROIC data saved to {csv_filename}")

    return roic_results

def get_plot_filenames():
    """Return a dictionary containing the filenames for the plot and CSV."""
    return {
        'plot_filename': '30yr_fixed_day_sip_subplot.png',
        'csv_filename': 'roic_vs_year_fixed_day.csv'
    }
