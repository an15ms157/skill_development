import pandas as pd
import numpy as np

def perform_sip_simulation(df, rolling_years=30, dma_window=500, investment_gap_days=30):
    """
    Perform a 30-year rolling window SIP simulation by investing 1 USD every 30 days only when 
    the price is below the 200-day moving average.

    Parameters:
    - df: DataFrame with S&P 500 data (columns: Date, Year, Index)
    - rolling_years: Number of years for each rolling window (default: 30)
    - dma_window: 200-day moving average (default: 200 days)
    - investment_gap_days: Interval between investments (default: 30 days)

    Returns: List of dictionaries with ROIC and window metadata.
    """
    print("Starting 30-Year Below 200-DMA SIP Simulation...")

    roic_results = []
    min_year = df['Year'].min()
    max_year = df['Year'].max() - rolling_years + 1

    for start_year in range(min_year, max_year + 1):
        end_year = start_year + rolling_years - 1
        df_window = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy().reset_index(drop=True)

        if df_window.empty:
            continue

        # Calculate 200-day moving average
        df_window['DMA_200'] = df_window['Index'].rolling(window=dma_window, min_periods=1).mean()
        df_window['DMA_200'] = df_window['DMA_200'].fillna(method='ffill')

        investment_dates = []
        investment_prices = []

        next_allowed_date = None
        started = False

        for i in range(len(df_window)):
            row = df_window.iloc[i]
            date = pd.to_datetime(row['Date'])
            price = row['Index']
            dma = row['DMA_200']

            # Ensure price and dma are scalar values
            price = float(price) if isinstance(price, (int, float)) else float(price.values[0])
            dma = float(dma) if isinstance(dma, (int, float)) else float(dma.values[0])

            # First valid investment
            if price < dma:
                if not started:
                    investment_dates.append(date)
                    investment_prices.append(price)
                    next_allowed_date = date + pd.Timedelta(days=investment_gap_days)
                    started = True
                elif next_allowed_date is not None and isinstance(next_allowed_date, pd.Timestamp) and date >= next_allowed_date:
                    # Subsequent investments
                    investment_dates.append(date)
                    investment_prices.append(price)
                    next_allowed_date = date + pd.Timedelta(days=investment_gap_days)

        if not investment_prices:
            continue

        units_bought = 1 / np.array(investment_prices)
        total_units = np.sum(units_bought)
        final_price = df_window['Index'].iloc[-1]
        final_price = float(final_price) if isinstance(final_price, (int, float)) else float(final_price.values[0])
        final_value = total_units * final_price
        total_invested = len(investment_prices)
        net_roic = (final_value - total_invested) / total_invested * 100

        roic_results.append({
            'StartYear': start_year,
            'EndYear': end_year,
            'NumInvestments': total_invested,
            'NetROIC': net_roic,
            'StartDate': df_window['Date'].iloc[0],
            'EndDate': df_window['Date'].iloc[-1],
            'IndexValues': df_window[['Date', 'Index']]  # Add IndexValues for plotting
        })

    return roic_results

def get_plot_filenames():
    return {
        'plot_filename': '30yr_below_200dma_sip_plot.png',
        'csv_filename': 'roic_below_200dma.csv'
    }
