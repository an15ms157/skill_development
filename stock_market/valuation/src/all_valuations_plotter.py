import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from dcf_monte_carlo import extract_dcf_inputs, run_monte_carlo, DCFModel
import logging
import sys
import os
from io import StringIO

# Ensure log directory exists
if not os.path.exists('log'):
    os.makedirs('log')

# Custom stream to redirect print statements to logging
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

# Set up logging to both console and file
def setup_logging(ticker):
    """Set up logging to both console and a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicate logs
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(f'log/{ticker}_valuations.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Redirect stdout to logger to capture print statements from libraries
    sys.stdout = StreamToLogger(logger, logging.INFO)

def load_data(ticker, data_dir="data"):
    """Load financial data from Excel files."""
    income_file = f"{data_dir}/{ticker}/{ticker} Income Statement (Annual) - Discounting Cash Flows.xlsx"
    balance_file = f"{data_dir}/{ticker}/{ticker} Balance Sheet Statement (Annual) - Discounting Cash Flows.xlsx"
    cashflow_file = f"{data_dir}/{ticker}/{ticker} Cash Flow Statement (Annual) - Discounting Cash Flows.xlsx"
    ratios_file = f"{data_dir}/{ticker}/{ticker} Financial Ratios (Annual) - Discounting Cash Flows.xlsx"

    try:
        income_df = pd.read_excel(income_file, sheet_name=0, header=0, index_col=0)
        balance_df = pd.read_excel(balance_file, sheet_name=0, header=0, index_col=0)
        cashflow_df = pd.read_excel(cashflow_file, sheet_name=0, header=0, index_col=0)
        ratios_df = pd.read_excel(ratios_file, sheet_name=0, header=0, index_col=0)

        # Clean index and column names
        for df in [income_df, balance_df, cashflow_df, ratios_df]:
            df.index = df.index.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
            df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
            df.columns = [col if col == 'LTM' else col.split(' ')[0].split('-')[0] for col in df.columns]
        
        logging.debug(f"Loaded data for {ticker}:")
        logging.debug(f"Income DF columns: {income_df.columns.tolist()}")
        logging.debug(f"Balance DF columns: {balance_df.columns.tolist()}")
        logging.debug(f"Cashflow DF columns: {cashflow_df.columns.tolist()}")
        logging.debug(f"Ratios DF columns: {ratios_df.columns.tolist()}")
        
        return income_df, balance_df, cashflow_df, ratios_df
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {e}")
        return None, None, None, None

def calculate_lynch_valuation(year, ratios_df, n_years=5):
    """Calculate Lynch valuation based on EPS and dividend yield."""
    try:
        # Ensure ratios_df has the expected structure
        if 'Year' not in ratios_df.columns:
            logging.debug(f"Transposing ratios_df for year {year}")
            ratios_df = ratios_df.T
            ratios_df['Year'] = pd.to_numeric(ratios_df.index.map(lambda x: '2025' if x == 'LTM' else x), errors='coerce')
            ratios_df = ratios_df.dropna(subset=['Year'])
            ratios_df['Year'] = ratios_df['Year'].astype(int)
        else:
            logging.debug(f"ratios_df already has 'Year' column for year {year}")

        logging.debug(f"ratios_df columns after processing: {ratios_df.columns.tolist()}")
        logging.debug(f"ratios_df head:\n{ratios_df.head().to_string()}")

        df_up_to_year = ratios_df[ratios_df['Year'] <= year]
        if len(df_up_to_year) < n_years + 1:
            logging.warning(f"Not enough data for Lynch valuation up to year {year}: {len(df_up_to_year)} rows available, need {n_years + 1}")
            return np.nan

        current_data = df_up_to_year[df_up_to_year['Year'] == year]
        if current_data.empty:
            logging.warning(f"No data available for year {year}")
            return np.nan

        logging.debug(f"Current data for year {year}:\n{current_data.to_string()}")
        
        # Extract EPS for the current year
        if 'Earnings Per Share' not in current_data.columns:
            logging.warning(f"'Earnings Per Share' not found in data for year {year}")
            return np.nan
        end_eps = pd.to_numeric(current_data['Earnings Per Share'].iloc[0], errors='coerce')
        
        start_year = year - n_years
        start_data = df_up_to_year[df_up_to_year['Year'] == start_year]
        if start_data.empty:
            logging.warning(f"No data available for start year {start_year}")
            return np.nan

        logging.debug(f"Start data for year {start_year}:\n{start_data.to_string()}")
        
        # Extract EPS for the start year
        if 'Earnings Per Share' not in start_data.columns:
            logging.warning(f"'Earnings Per Share' not found in data for start year {start_year}")
            return np.nan
        start_eps = pd.to_numeric(start_data['Earnings Per Share'].iloc[0], errors='coerce')

        if pd.isna(end_eps) or pd.isna(start_eps) or start_eps <= 0 or end_eps <= 0:
            logging.warning(f"Invalid EPS values for Lynch valuation in year {year}: start_eps={start_eps}, end_eps={end_eps}")
            return np.nan

        cagr = ((end_eps / start_eps) ** (1 / n_years) - 1) * 100
        if cagr < 0:
            logging.warning(f"Negative CAGR for Lynch valuation in year {year}: {cagr}")
            return np.nan

        dividend_yield = pd.to_numeric(current_data.get('Annual Dividend Yield', pd.Series([0])).iloc[0], errors='coerce') / 100
        lynch_price = end_eps * (cagr + dividend_yield * 100)
        logging.debug(f"Lynch valuation for {year}: EPS={end_eps}, CAGR={cagr}, Dividend Yield={dividend_yield}, Lynch Price={lynch_price}")
        return lynch_price
    except Exception as e:
        logging.error(f"Error calculating Lynch valuation for {year}: {e}")
        return np.nan

def calculate_fcv_valuation(year, income_df, balance_df, years_for_average=5):
    """Calculate FCV (EPV) valuation."""
    try:
        available_years = [col for col in income_df.columns if col.isdigit() and int(col) <= year]
        if len(available_years) < years_for_average:
            logging.warning(f"Not enough years for FCV valuation up to {year}: {len(available_years)} available, need {years_for_average}")
            return np.nan

        past_years = sorted(available_years, reverse=True)[:years_for_average]
        logging.debug(f"Using years {past_years} for FCV valuation in {year}")
        
        if 'Net Income' not in income_df.index:
            logging.warning(f"'Net Income' not found in income_df for year {year}")
            return np.nan
        net_income = pd.to_numeric(income_df.loc['Net Income', past_years], errors='coerce').mean()
        
        wacc = 0.07  # Fixed WACC
        if 'Net Debt' not in balance_df.index or str(year) not in balance_df.columns:
            logging.warning(f"'Net Debt' or year {year} not found in balance_df")
            return np.nan
        net_debt = pd.to_numeric(balance_df.loc['Net Debt', str(year)], errors='coerce')
        
        if 'Diluted Weighted Average Shares Outstanding' not in income_df.index or str(year) not in income_df.columns:
            logging.warning(f"'Diluted Weighted Average Shares Outstanding' or year {year} not found in income_df")
            return np.nan
        shares = pd.to_numeric(income_df.loc['Diluted Weighted Average Shares Outstanding', str(year)], errors='coerce')

        if pd.isna(net_income) or pd.isna(net_debt) or pd.isna(shares) or shares <= 0:
            logging.warning(f"Invalid data for FCV valuation in year {year}: net_income={net_income}, net_debt={net_debt}, shares={shares}")
            return np.nan

        epv = (net_income / wacc - net_debt) / shares
        logging.debug(f"FCV valuation for {year}: net_income={net_income}, net_debt={net_debt}, shares={shares}, EPV={epv}")
        return epv
    except Exception as e:
        logging.error(f"Error calculating FCV valuation for {year}: {e}")
        return np.nan

def plot_all_valuations(ticker, start_year=2000, end_year=2024, data_dir="data"):
    """Plot Lynch, FCV, and DCF valuations with stock price and SMA."""
    # Determine currency
    currency = 'INR' if ticker.endswith('.NS') else 'USD'

    # Load data
    income_df, balance_df, cashflow_df, ratios_df = load_data(ticker, data_dir)
    if income_df is None:
        logging.error(f"Cannot proceed for {ticker} due to data loading failure.")
        return None

    # Initialize valuation storage
    valuations = {
        'year': [],
        'lynch': [],
        'fcv': [],
        'simple_mean': [],
        'simple_sigma2_low': [],
        'simple_sigma2_high': [],
        'enhanced_mean': [],
        'enhanced_sigma2_low': [],
        'enhanced_sigma2_high': []
    }

    # Calculate valuations for each year
    for year in range(start_year, end_year + 1):
        logging.info(f"Processing valuations for {ticker} in year {year}...")
        
        # Lynch Valuation
        lynch_val = calculate_lynch_valuation(year, ratios_df.copy())
        
        # FCV Valuation
        fcv_val = calculate_fcv_valuation(year, income_df, balance_df)
        
        # DCF Valuation
        dcf_inputs = extract_dcf_inputs(ticker, data_dir=data_dir, start_year=year, end_year=year)
        simple_mean = simple_sigma2_low = simple_sigma2_high = np.nan
        enhanced_mean = enhanced_sigma2_low = enhanced_sigma2_high = np.nan
        
        if dcf_inputs and not pd.isna(dcf_inputs['starting_fcf']) and not pd.isna(dcf_inputs['shares_outstanding']):
            if dcf_inputs['starting_fcf'] <= 0:
                logging.warning(f"Skipping DCF for {year}: Negative or zero starting FCF ({dcf_inputs['starting_fcf']})")
            else:
                try:
                    # Simple DCF
                    simple_model = DCFModel(
                        starting_fcf=dcf_inputs['starting_fcf'],
                        wacc=dcf_inputs['wacc'],
                        high_growth_rate=dcf_inputs['high_growth_rate'],
                        terminal_growth_rate=dcf_inputs['terminal_growth_rate'],
                        net_debt=dcf_inputs['net_debt'],
                        shares_outstanding=dcf_inputs['shares_outstanding'],
                        use_enhanced_model=False,
                        roic=dcf_inputs['roic'],
                        ticker=ticker,
                        reinvestment_cap=dcf_inputs['reinvestment_cap']
                    )
                    simple_results = run_monte_carlo(simple_model, num_simulations=1000)
                    if len(simple_results) > 0:
                        simple_mean = np.mean(simple_results)
                        simple_std = np.std(simple_results)
                        simple_sigma2_low = max(simple_mean - 2 * simple_std, 0)
                        simple_sigma2_high = simple_mean + 2 * simple_std

                    # Enhanced DCF
                    enhanced_model = DCFModel(
                        starting_fcf=dcf_inputs['starting_fcf'],
                        wacc=dcf_inputs['wacc'],
                        high_growth_rate=dcf_inputs['high_growth_rate'],
                        terminal_growth_rate=dcf_inputs['terminal_growth_rate'],
                        net_debt=dcf_inputs['net_debt'],
                        shares_outstanding=dcf_inputs['shares_outstanding'],
                        use_enhanced_model=True,
                        roic=dcf_inputs['roic'],
                        ticker=ticker,
                        reinvestment_cap=dcf_inputs['reinvestment_cap']
                    )
                    enhanced_results = run_monte_carlo(enhanced_model, num_simulations=1000)
                    if len(enhanced_results) > 0:
                        enhanced_mean = np.mean(enhanced_results)
                        enhanced_std = np.std(enhanced_results)
                        enhanced_sigma2_low = max(enhanced_mean - 2 * enhanced_std, 0)
                        enhanced_sigma2_high = enhanced_mean + 2 * enhanced_std
                except Exception as e:
                    logging.error(f"Error calculating DCF for {year}: {e}")

        # Store valuations
        valuations['year'].append(year)
        valuations['lynch'].append(lynch_val)
        valuations['fcv'].append(fcv_val)
        valuations['simple_mean'].append(simple_mean)
        valuations['simple_sigma2_low'].append(simple_sigma2_low)
        valuations['simple_sigma2_high'].append(simple_sigma2_high)
        valuations['enhanced_mean'].append(enhanced_mean)
        valuations['enhanced_sigma2_low'].append(enhanced_sigma2_low)
        valuations['enhanced_sigma2_high'].append(enhanced_sigma2_high)

    # Fetch stock price data
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        stock_data_weekly = stock_data['Close'].resample('W-FRI').last()
        # Calculate 200-week SMA on weekly data
        stock_data_weekly_200sma = stock_data_weekly.rolling(window=200).mean()
    except Exception as e:
        logging.error(f"Error fetching stock price data for {ticker}: {e}")
        stock_data_weekly = pd.Series()
        stock_data_weekly_200sma = pd.Series()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot stock price and 200-week SMA
    if not stock_data_weekly.empty:
        ax1.plot(stock_data_weekly.index, stock_data_weekly, label='Weekly Stock Price', color='blue')
        ax1.plot(stock_data_weekly_200sma.index, stock_data_weekly_200sma, label='200-week SMA', color='yellow', linestyle='--')
    
    # Plot valuations
    valuation_dates = [datetime(y, 12, 31) for y in valuations['year']]
    ax1.plot(valuation_dates, valuations['lynch'], label='Lynch Valuation', color='green', marker='o')
    ax1.plot(valuation_dates, valuations['fcv'], label='FCV (EPV)', color='purple', marker='s')
    
    # Plot DCF Simple and Enhanced as lines with shaded bands for ±2σ
    ax1.plot(valuation_dates, valuations['simple_mean'], label='DCF Simple Mean', color='red', marker='o')
    ax1.fill_between(valuation_dates, valuations['simple_sigma2_low'], valuations['simple_sigma2_high'],
                     color='red', alpha=0.2, label='DCF Simple ±2σ')
    
    ax1.plot(valuation_dates, valuations['enhanced_mean'], label='DCF Enhanced Mean', color='brown', marker='o')
    ax1.fill_between(valuation_dates, valuations['enhanced_sigma2_low'], valuations['enhanced_sigma2_high'],
                     color='brown', alpha=0.2, label='DCF Enhanced ±2σ')

    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Price / Valuation ({currency})')
    ax1.set_yscale('log')
    ax1.set_title(f'{ticker} Valuations and Stock Price')
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_all_valuations_log.png')
    plt.close()
    logging.info(f"Plot saved as 'plots/{ticker}_all_valuations_log.png'")
    
    return valuations

if __name__ == "__main__":
    ticker = "AMZN"
    start_year = 2000
    end_year = 2024
    
    # Set up logging
    setup_logging(ticker)
    
    valuations = plot_all_valuations(ticker, start_year=start_year, end_year=end_year)
    
    if valuations:
        # Calculate time averages for each valuation method
        logging.info(f"\nTime-Averaged Valuations for {ticker} ({start_year}-{end_year}):")
        for val_type in ['lynch', 'fcv', 'simple_mean', 'enhanced_mean']:
            valid_vals = [v for v in valuations[val_type] if not pd.isna(v)]
            if valid_vals:
                avg_val = np.mean(valid_vals)
                logging.info(f"{val_type.replace('_mean', ' DCF').title()}: {avg_val:.2f} USD")
            else:
                logging.info(f"{val_type.replace('_mean', ' DCF').title()}: No valid data")