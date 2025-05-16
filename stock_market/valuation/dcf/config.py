import logging
import os
import pandas as pd
import numpy as np
import json

# Setup logging
LOG_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, 'default.log')

logging.basicConfig(
    filename=DEFAULT_LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()

def setup_log_file(ticker):
    """Update log file name dynamically based on ticker."""
    log_file = os.path.join(LOG_DIR, f"{ticker}.log")
    logger.handlers[0].baseFilename = log_file

def detect_units(df, ticker, context=""):
    """Detect scaling factors in a DataFrame based on column or index labels."""
    scaling_factors = []
    for col in df.columns:
        col_str = str(col).lower()
        if 'million' in col_str or 'mm' in col_str:
            scaling_factors.append(1e6)
        elif 'thousand' in col_str or 'k' in col_str:
            scaling_factors.append(1e3)
        elif 'billion' in col_str or 'bn' in col_str:
            scaling_factors.append(1e9)
    for index in df.index:
        index_str = str(index).lower()
        if 'million' in index_str or 'mm' in index_str:
            scaling_factors.append(1e6)
        elif 'thousand' in index_str or 'k' in index_str:
            scaling_factors.append(1e3)
        elif 'billion' in index_str or 'bn' in col_str:
            scaling_factors.append(1e9)
    for _, row in df.head(5).iterrows():
        for value in row:
            value_str = str(value).lower()
            if 'million' in value_str or 'mm' in value_str:
                scaling_factors.append(1e6)
            elif 'thousand' in value_str or 'k' in value_str:
                scaling_factors.append(1e3)
            elif 'billion' in value_str or 'bn' in value_str:
                scaling_factors.append(1e9)
    if scaling_factors:
        scaling_factor = max(scaling_factors)
        print(f"Detected units for {ticker} {context}: scaling factor {scaling_factor}")
        return scaling_factor
    print(f"No unit indicators found for {ticker} {context}. Assuming units in millions.")
    return 1e6

def extract_dcf_inputs(ticker, data_dir="/home/abhishek/Desktop/skill_development/stock_market/valuation/data", start_year=1950, end_year=2024):
    try:
        income_file = f"{data_dir}/{ticker}/{ticker} Income Statement (Annual) - Discounting Cash Flows.xlsx"
        balance_file = f"{data_dir}/{ticker}/{ticker} Balance Sheet Statement (Annual) - Discounting Cash Flows.xlsx"
        cashflow_file = f"{data_dir}/{ticker}/{ticker} Cash Flow Statement (Annual) - Discounting Cash Flows.xlsx"
        ratios_file = f"{data_dir}/{ticker}/{ticker} Financial Ratios (Annual) - Discounting Cash Flows.xlsx"

        income_df = pd.read_excel(income_file, sheet_name=0, header=0, index_col=0)
        balance_df = pd.read_excel(balance_file, sheet_name=0, header=0, index_col=0)
        cashflow_df = pd.read_excel(cashflow_file, sheet_name=0, header=0, index_col=0)
        ratios_df = pd.read_excel(ratios_file, sheet_name=0, header=0, index_col=0)
        print(f"Successfully read Excel files for {ticker}")

        income_units = detect_units(income_df, ticker, "Income Statement")
        balance_units = detect_units(balance_df, ticker, "Balance Sheet")
        cashflow_units = detect_units(cashflow_df, ticker, "Cash Flow Statement")
        ratios_units = detect_units(ratios_df, ticker, "Financial Ratios")

        financial_scaling_factors = [income_units, balance_units, cashflow_units]
        financial_scaling_factor = max(financial_scaling_factors)
        print(f"Using consistent financial scaling factor for {ticker}: {financial_scaling_factor}")
        income_units = financial_scaling_factor
        balance_units = financial_scaling_factor
        cashflow_units = financial_scaling_factor

        for df in [income_df, balance_df, cashflow_df, ratios_df]:
            df.index = df.index.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
            df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
            df.columns = [col if col == 'LTM' else col.split(' ')[0].split('-')[0] for col in df.columns]

        def find_row_label(df, target_labels):
            for row in df.index:
                for target_label in target_labels:
                    if target_label.lower() in row.lower():
                        return row
            return None

        row_labels = {
            'Free Cash Flow': ['Free Cash Flow'],
            'Net Debt': ['Net Debt'],
            'Weighted Average Shares Outstanding': ['Weighted Average Shares Outstanding'],
            'Price Per Share': ['Price Per Share'],
            'Total Debt': ['Total Debt'],
            'Total Equity': ['Total Equity', "Total Stockholders' Equity"],
            'Income Before Tax': ['Income Before Tax', 'Profit Before Tax'],
            'Income Tax Expense': ['Income Tax Expense', 'Tax Expense'],
            'Effective Tax Rate': ['Effective Tax Rate'],
            'Revenue': ['Revenue', 'Total Revenue'],
            'Interest Expense': ['Interest Expense', 'Interest Expense (Income)'],
            'Operating Profit': ['Operating Profit', 'Operating Income', 'EBIT'],
            'Cash and Cash Equivalents': ['Cash and Cash Equivalents', 'Cash & Equivalents', 'Cash', 'Cash and Short Term Investments'],
            'Capital Expenditure': ['Capital Expenditure', 'Capital Expenditures', 'Capex'],
            'Change in Working Capital': ['Change in Working Capital', 'Changes in Working Capital']
        }
        row_to_df = {
            'Free Cash Flow': cashflow_df,
            'Net Debt': balance_df,
            'Weighted Average Shares Outstanding': income_df,
            'Price Per Share': ratios_df,
            'Total Debt': balance_df,
            'Total Equity': balance_df,
            'Income Before Tax': income_df,
            'Income Tax Expense': income_df,
            'Effective Tax Rate': ratios_df,
            'Revenue': income_df,
            'Interest Expense': income_df,
            'Operating Profit': income_df,
            'Cash and Cash Equivalents': balance_df,
            'Capital Expenditure': cashflow_df,
            'Change in Working Capital': cashflow_df
        }

        found_labels = {}
        for key, targets in row_labels.items():
            found_labels[key] = find_row_label(row_to_df[key], targets)
            if not found_labels[key] and key in ['Free Cash Flow', 'Weighted Average Shares Outstanding']:
                print(f"Error: Missing required row '{targets}' in {key.lower().replace(' ', '_')} DataFrame")
                return None
            elif not found_labels[key]:
                print(f"Warning: Row '{targets}' not found. Using default for {key}.")

        print("Found rows:", {k: v for k, v in found_labels.items() if v})

        all_years = [col for col in income_df.columns if col.isdigit()]
        all_years.sort(reverse=True)
        if not all_years:
            print("Error: No years found in Income Statement columns")
            return None

        available_years = [col for col in all_years if start_year <= int(col) <= end_year]
        if not available_years:
            print(f"No data for years {start_year} to {end_year}. Falling back to latest available year: {all_years[0]}")
            available_years = [all_years[0]]
        
        available_years.sort(reverse=True)
        latest_year = available_years[0]
        previous_year = available_years[1] if len(available_years) > 1 else None

        def convert_to_numeric(value, is_shares=False, is_price=False, scaling_factor=1):
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
                    return float(value) * scaling_factor
                except ValueError:
                    return np.nan
            elif isinstance(value, (int, float)):
                return float(value) * scaling_factor
            return np.nan

        def get_value(df, row, year, prev_year=None, default=None, is_shares=False, is_price=False, scaling_factor=1):
            if row not in df.index:
                print(f"Warning: {row} not in DataFrame index. Using default: {default}")
                return default
            raw_value = df.loc[row, year]
            print(f"Raw {row} for {year}: {raw_value}")
            value = convert_to_numeric(raw_value, is_shares=is_shares, is_price=is_price, scaling_factor=scaling_factor)
            print(f"Converted {row} for {year} with scaling factor {scaling_factor}: {value}")
            if pd.notna(value):
                return value
            print(f"Warning: {row} is NaN for {year}. Trying previous year.")
            if prev_year and prev_year in df.columns:
                raw_value = df.loc[row, prev_year]
                print(f"Raw {row} for {prev_year}: {raw_value}")
                value = convert_to_numeric(raw_value, is_shares=is_shares, is_price=is_price, scaling_factor=scaling_factor)
                print(f"Converted {row} for {prev_year} with scaling factor {scaling_factor}: {value}")
                if pd.notna(value):
                    print(f"Using {row} from {prev_year}: {value}")
                    return value
            print(f"Warning: {row} not found in {year} or {prev_year}. Using default: {default}")
            return default

        starting_fcf = get_value(cashflow_df, found_labels['Free Cash Flow'], latest_year, previous_year, default=1000e6, scaling_factor=cashflow_units)
        net_debt = get_value(balance_df, found_labels['Net Debt'], latest_year, previous_year, default=0, scaling_factor=balance_units)
        shares_scaling_factor = detect_units(income_df, ticker, "Shares Outstanding")
        shares_outstanding = get_value(income_df, found_labels['Weighted Average Shares Outstanding'], latest_year, previous_year, default=100e6, is_shares=True, scaling_factor=shares_scaling_factor)
        current_price = get_value(ratios_df, found_labels['Price Per Share'], latest_year, previous_year, default=100, is_price=True, scaling_factor=1)
        total_debt = get_value(balance_df, found_labels['Total Debt'], latest_year, previous_year, default=0, scaling_factor=balance_units)
        total_equity = get_value(balance_df, found_labels['Total Equity'], latest_year, previous_year, default=1000e6, scaling_factor=balance_units)
        interest_expense = abs(get_value(income_df, found_labels['Interest Expense'], latest_year, previous_year, default=0, scaling_factor=income_units))
        income_before_tax = get_value(income_df, found_labels['Income Before Tax'], latest_year, previous_year, default=None, scaling_factor=income_units)
        income_tax_expense = get_value(income_df, found_labels['Income Tax Expense'], latest_year, previous_year, default=None, scaling_factor=income_units)
        operating_profit = get_value(income_df, found_labels['Operating Profit'], latest_year, previous_year, default=0, scaling_factor=income_units)
        cash = get_value(balance_df, found_labels['Cash and Cash Equivalents'], latest_year, previous_year, default=0, scaling_factor=balance_units)
        revenue = get_value(income_df, found_labels['Revenue'], latest_year, previous_year, default=0, scaling_factor=income_units)
        capex = get_value(cashflow_df, found_labels['Capital Expenditure'], latest_year, previous_year, default=0, scaling_factor=cashflow_units)
        working_capital_change = get_value(cashflow_df, found_labels['Change in Working Capital'], latest_year, previous_year, default=0, scaling_factor=cashflow_units)

        margin_compression = 0.005
        if found_labels['Operating Profit'] and found_labels['Revenue']:
            margin_years = [y for y in all_years if start_year <= int(y) <= end_year]
            margin_years.sort()
            if len(margin_years) >= 2:
                margins = []
                for year in margin_years:
                    op_profit = get_value(income_df, found_labels['Operating Profit'], year, None, default=0, scaling_factor=income_units)
                    rev = get_value(income_df, found_labels['Revenue'], year, None, default=1, scaling_factor=income_units)
                    if rev != 0 and pd.notna(op_profit) and pd.notna(rev):
                        margin = op_profit / rev
                        margins.append(margin)
                    else:
                        margins.append(np.nan)
                margin_changes = [margins[i] - margins[i+1] for i in range(len(margins)-1) if pd.notna(margins[i]) and pd.notna(margins[i+1])]
                negative_changes = [abs(change) for change in margin_changes if change < 0]
                if negative_changes:
                    margin_compression = np.mean(negative_changes)
                    print(f"Calculated margin compression: {margin_compression:.4f}")
                else:
                    print(f"No margin compression detected. Using default: {margin_compression}")
            else:
                print(f"Insufficient years for margin compression. Using default: {margin_compression}")
        else:
            print(f"Missing Operating Profit or Revenue. Using default margin compression: {margin_compression}")

        try:
            tax_rate = get_value(ratios_df, found_labels['Effective Tax Rate'], latest_year, previous_year, default=None, scaling_factor=1)
            if pd.isna(tax_rate):
                raise ValueError("Effective Tax Rate is NaN")
            print(f"Effective Tax Rate: {tax_rate:.2%}")
        except (KeyError, ValueError):
            print(f"Could not use Effective Tax Rate.")
            try:
                if pd.notna(income_before_tax) and pd.notna(income_tax_expense) and income_before_tax != 0:
                    tax_rate = income_tax_expense / income_before_tax
                    print(f"Calculated tax rate: {tax_rate:.2%}")
                else:
                    print(f"Invalid data for tax rate. Using default: 21.0%")
                    tax_rate = 0.21
            except (KeyError, TypeError) as e:
                print(f"Could not calculate tax rate: {e}. Using default: 21.0%")
                tax_rate = 0.21

        nopat = operating_profit * (1 - tax_rate) if pd.notna(operating_profit) and pd.notna(tax_rate) else 0
        invested_capital = total_equity + total_debt - cash if pd.notna(total_equity) and pd.notna(total_debt) and pd.notna(cash) else 1e9
        roic = nopat / invested_capital if invested_capital != 0 and pd.notna(nopat) else 0.15
        print(f"Calculated ROIC: {roic:.2%}")

        risk_free_rate = 0.045
        market_return = 0.09
        beta = 0.9
        terminal_growth_rate = 0.025
        growth_cap = 0.08
        reinvestment_cap = 0.05
        currency = 'USD'

        if ticker == "NKE":
            beta = 0.9
            terminal_growth_rate = 0.025

        wacc_MIN = 0.04
        wacc_MAX = 0.15
        market_equity = shares_outstanding * current_price if pd.notna(shares_outstanding) and pd.notna(current_price) else total_equity
        cost_of_debt_pretax = interest_expense / total_debt if total_debt != 0 and interest_expense != 0 else 0.05
        cost_of_debt = cost_of_debt_pretax * (1 - tax_rate) if cost_of_debt_pretax != 0 else 0.05 * (1 - tax_rate)
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        total_capital = total_debt + market_equity if pd.notna(total_debt) and pd.notna(market_equity) else market_equity
        debt_weight = total_debt / total_capital if total_capital != 0 else 0
        equity_weight = market_equity / total_capital if total_capital != 0 else 1
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt)
        wacc = min(max(wacc, wacc_MIN), wacc_MAX) if pd.notna(wacc) else 0.08
        print(f"Calculated WACC: {wacc:.2%}")

        revenue_years = [y for y in all_years if int(y) <= start_year]
        revenue_years.sort()
        if len(revenue_years) >= 2 and found_labels['Revenue']:
            revenue_end = get_value(income_df, found_labels['Revenue'], revenue_years[-1], None, default=0, scaling_factor=income_units)
            revenue_start = get_value(income_df, found_labels['Revenue'], revenue_years[0], None, default=0, scaling_factor=income_units)
            years_span = int(revenue_years[-1]) - int(revenue_years[0])
            cagr = (revenue_end / revenue_start) ** (1 / years_span) - 1 if pd.notna(revenue_start) and revenue_start != 0 and years_span > 0 else 0.05
            print(f"Revenue CAGR: {cagr:.2%}")
        else:
            cagr = 0.05
        high_growth_rate = min(max(cagr, 0.05), growth_cap) if pd.notna(cagr) else 0.05

        CONFIG = {
            'ticker': ticker,
            'data_dir': data_dir,
            'start_year': start_year,
            'end_year': end_year,
            'currency': currency,
            'default_fcf': starting_fcf,
            'default_net_debt': net_debt,
            'default_shares_outstanding': shares_outstanding,
            'default_price_per_share': current_price,
            'default_total_debt': total_debt,
            'default_total_equity': total_equity,
            'default_tax_rate': tax_rate,
            'default_operating_profit': operating_profit,
            'default_cash': cash,
            'default_revenue': revenue,
            'default_nopat': nopat,
            'default_roic': roic,
            'default_capex': capex,
            'default_working_capital_change': working_capital_change,
            'default_beta': beta,
            'default_wacc': wacc,
            'risk_free_rate': risk_free_rate,
            'market_return': market_return,
            'gdp_growth': terminal_growth_rate,
            'high_growth_years': 10,
            'high_growth_rate': high_growth_rate,
            'growth_cap': growth_cap,
            'reinvestment_cap': reinvestment_cap,
            'wacc_min': wacc_MIN,
            'wacc_max': wacc_MAX,
            'fcf_volatility': 0.3,
            'growth_decay': 0.015,
            'default_margin_compression': margin_compression,
            'default_wacc_drift_std': 0.002,
            'num_simulations': 50,
            'fcf_error': 0.10,
            'net_debt_error': 0.05,
            'volatility_factor': 0.05,
            'RUNS': 10,
            'tolerance_level': 0.05,
            'parameter_bounds': [
                (0.035, 0.20),
                (0.01, high_growth_rate*3),
                (0.00, 0.035),
                (0.0, high_growth_rate*3),
                (0.0, margin_compression*3),
                (0.0, 0.80),
                (0.0, 10.0)
            ]
        }

        config_folder = os.path.join('config_file')
        os.makedirs(config_folder, exist_ok=True)
        output_file = os.path.join(config_folder, f'{ticker}_config.json')
        with open(output_file, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        print(f"CONFIG saved to {output_file}")

        logger.info(f"WACC calculation for {ticker}:")
        logger.info(f"Market Equity: {market_equity:.2f}")
        logger.info(f"Total Debt: {total_debt:.2f}")
        logger.info(f"Cost of Equity: {cost_of_equity:.2%}")
        logger.info(f"Cost of Debt: {cost_of_debt:.2%}")
        logger.info(f"Equity Weight: {equity_weight:.2%}")
        logger.info(f"Debt Weight: {debt_weight:.2%}")
        logger.info(f"Tax Rate: {tax_rate:.2%}")
        logger.info(f"WACC: {wacc:.2%}")

        return CONFIG

    except Exception as e:
        print(f"Error processing data for {ticker}: {e}")
        logger.error(f"Error processing data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    CONFIG = extract_dcf_inputs(ticker='NKE', data_dir='data', start_year=1950, end_year=2024)
    if CONFIG and 'ticker' in CONFIG:
        log_file = os.path.join(LOG_DIR, f"{CONFIG['ticker']}.log")
        logger.handlers[0].baseFilename = log_file