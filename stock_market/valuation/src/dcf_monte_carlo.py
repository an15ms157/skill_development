import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
use_enhanced_model = True

class DCFModel:
    def __init__(self, 
                 starting_fcf, 
                 wacc, 
                 high_growth_rate, 
                 terminal_growth_rate, 
                 net_debt, 
                 shares_outstanding,
                 high_growth_years=10,
                 growth_decay=0.001 if use_enhanced_model else 0,
                 margin_compression=0.002 if use_enhanced_model else 0,
                 roic=None,
                 wacc_drift_std=0.002 if use_enhanced_model else 0,
                 use_enhanced_model=use_enhanced_model,
                 ticker="",
                 reinvestment_cap=0.1):
        self.starting_fcf = starting_fcf
        self.wacc = wacc
        self.high_growth_rate = high_growth_rate
        self.terminal_growth_rate = terminal_growth_rate
        self.net_debt = net_debt
        self.shares_outstanding = shares_outstanding
        self.high_growth_years = high_growth_years
        self.growth_decay = growth_decay
        self.margin_compression = margin_compression
        self.roic = roic
        self.wacc_drift_std = wacc_drift_std
        self.use_enhanced_model = use_enhanced_model
        self.ticker = ticker
        self.reinvestment_cap = reinvestment_cap
        
        self.validate_inputs()
    
    def validate_inputs(self):
        if not pd.notna(self.wacc) or self.wacc <= self.terminal_growth_rate:
            raise ValueError(f"Invalid WACC: {self.wacc} (must be > terminal growth rate {self.terminal_growth_rate})")
        if not pd.notna(self.shares_outstanding) or self.shares_outstanding <= 0:
            raise ValueError(f"Invalid Shares Outstanding: {self.shares_outstanding} (must be positive)")
        if not pd.notna(self.starting_fcf) or self.starting_fcf <= 0:
            raise ValueError(f"Invalid Starting FCF: {self.starting_fcf} (must be positive)")
    
    def project_fcfs(self):
        fcfs = []
        fcf = self.starting_fcf
        growth = self.high_growth_rate
        wacc = self.wacc
        for _ in range(self.high_growth_years):
            if self.use_enhanced_model:
                margin_comp = np.random.uniform(0, self.margin_compression)
                fcf *= (1 - margin_comp)
                if self.roic is not None and self.roic != 0:
                    reinvestment_rate = (growth / self.roic) * 0.5
                    reinvestment_rate = min(max(reinvestment_rate, 0), self.reinvestment_cap)
                    fcf = fcf * (1 - reinvestment_rate)
                growth = max(growth - self.growth_decay, 0.03)
                fcf = fcf * (1 + growth)
                wacc += np.random.normal(0, self.wacc_drift_std)
                wacc = min(max(wacc, 0.05), 0.12)
            else:
                fcf = fcf * (1 + self.high_growth_rate)
            fcfs.append(fcf)
        return fcfs
    
    def calculate_terminal_value(self, final_fcf):
        return final_fcf * (1 + self.terminal_growth_rate) / (self.wacc - self.terminal_growth_rate)
    
    def calculate_dcf(self):
        fcfs = self.project_fcfs()
        terminal_value = self.calculate_terminal_value(fcfs[-1])
        
        pv_fcfs = sum([fcf / (1 + self.wacc) ** (i + 1) for i, fcf in enumerate(fcfs)])
        pv_terminal = terminal_value / (1 + self.wacc) ** self.high_growth_years
        
        enterprise_value = pv_fcfs + pv_terminal
        equity_value = max(enterprise_value - self.net_debt, 0)
        fair_value_per_share = equity_value / self.shares_outstanding if equity_value > 0 else 0
        
        return fair_value_per_share

def run_monte_carlo(base_model, num_simulations=10000, 
                    wacc_range=(0.05, 0.12), 
                    growth_range=(0.05, 0.12),
                    terminal_growth_range=(0.02, 0.04),
                    fcf_error=0.1,
                    net_debt_error=0.05):
    results = []
    failed_simulations = 0
    for _ in range(num_simulations):
        try:
            if base_model.use_enhanced_model:
                wacc = np.random.normal(base_model.wacc, 0.015)
                wacc = max(wacc, 0.05)
                high_growth = np.random.normal(base_model.high_growth_rate, 0.02)
                terminal_growth = np.random.normal(base_model.terminal_growth_rate, 0.005)
                net_debt_adjustment = np.random.normal(0, abs(base_model.net_debt) * net_debt_error)
                net_debt = base_model.net_debt + net_debt_adjustment
                net_debt = max(net_debt, 0)
            else:
                wacc = np.random.uniform(*wacc_range)
                high_growth = np.random.uniform(*growth_range)
                terminal_growth = np.random.uniform(*terminal_growth_range)
                net_debt = base_model.net_debt

            fcf_random = np.random.normal(base_model.starting_fcf, base_model.starting_fcf * fcf_error)

            terminal_growth = min(terminal_growth, wacc - 0.02)

            model = DCFModel(
                starting_fcf=max(fcf_random, 0.01),
                wacc=wacc,
                high_growth_rate=high_growth,
                terminal_growth_rate=terminal_growth,
                net_debt=net_debt,
                shares_outstanding=base_model.shares_outstanding,
                high_growth_years=base_model.high_growth_years,
                use_enhanced_model=base_model.use_enhanced_model,
                growth_decay=base_model.growth_decay,
                margin_compression=base_model.margin_compression,
                roic=base_model.roic,
                wacc_drift_std=base_model.wacc_drift_std,
                ticker=base_model.ticker,
                reinvestment_cap=base_model.reinvestment_cap
            )
            fair_value = model.calculate_dcf()
            if pd.notna(fair_value) and fair_value >= 0:
                results.append(fair_value)
        except Exception as e:
            failed_simulations += 1
            if _ < 10:
                print(f"Simulation {_} failed: {e}")
    
    if failed_simulations > 0:
        print(f"Warning: {failed_simulations} out of {num_simulations} simulations failed.")
    
    return np.array(results)

def plot_simulation_results(simulated_values, current_price=None, ticker="Stock", currency="USD"):
    if len(simulated_values) == 0:
        print(f"Error: No valid simulation results for {ticker}. Check input data.")
        return
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.histplot(simulated_values, bins=100, kde=True, color='skyblue')
    
    mean_val = np.mean(simulated_values)
    std_dev = np.std(simulated_values)
    
    plt.axvline(mean_val, color='green', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_dev, color='orange', linestyle='--', label=f'+1σ: {mean_val + std_dev:.2f}')
    plt.axvline(mean_val - std_dev, color='orange', linestyle='--', label=f'-1σ: {mean_val - std_dev:.2f}')
    plt.axvline(mean_val + 2*std_dev, color='red', linestyle='--', label=f'+2σ: {mean_val + 2*std_dev:.2f}')
    plt.axvline(mean_val - 2*std_dev, color='red', linestyle='--', label=f'-2σ: {mean_val - 2*std_dev:.2f}')
    plt.axvline(mean_val + 3*std_dev, color='purple', linestyle='--', label=f'+3σ: {mean_val + 3*std_dev:.2f}')
    plt.axvline(mean_val - 3*std_dev, color='purple', linestyle='--', label=f'-3σ: {mean_val - 3*std_dev:.2f}')
    
    if current_price and pd.notna(current_price):
        plt.axvline(current_price, color='black', linestyle='-', label=f'Current Price: {current_price:.2f}')
    
    plt.title(f'DCF Monte Carlo Simulation Results for {ticker}')
    plt.xlabel(f'Fair Value per Share ({currency})')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{ticker}_dcf_monte_carlo.png")
    plt.show()
    
    print(f"Mean Fair Value: {mean_val:.2f}")
    print(f"1σ Range: {mean_val - std_dev:.2f} to {mean_val + std_dev:.2f}")
    print(f"2σ Range: {mean_val - 2*std_dev:.2f} to {mean_val + 2*std_dev:.2f}")
    print(f"3σ Range: {mean_val - 3*std_dev:.2f} to {mean_val + 3*std_dev:.2f}")

def detect_units(df, ticker, context=""):
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
        elif 'billion' in index_str or 'bn' in index_str:
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
    print(f"No unit indicators found for {ticker} {context}. Assuming units in millions for {ticker}.")
    return 1e6

def extract_dcf_inputs(ticker, data_dir="data", start_year=2015, end_year=2025):
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
            'Net Non-Operating Interest': ['Net Non-Operating Interest', 'Interest Expense'],
            'Operating Profit': ['Operating Profit', 'Operating Income', 'EBIT'],
            'Cash and Cash Equivalents': ['Cash and Cash Equivalents', 'Cash & Equivalents', 'Cash', 'Cash and Short Term Investments']
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
            'Net Non-Operating Interest': income_df,
            'Operating Profit': income_df,
            'Cash and Cash Equivalents': balance_df
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
        income_before_tax = get_value(income_df, found_labels['Income Before Tax'], latest_year, previous_year, default=None, scaling_factor=income_units)
        income_tax_expense = get_value(income_df, found_labels['Income Tax Expense'], latest_year, previous_year, default=None, scaling_factor=income_units)
        net_non_operating_interest = get_value(income_df, found_labels['Net Non-Operating Interest'], latest_year, previous_year, default=0, scaling_factor=income_units)
        operating_profit = get_value(income_df, found_labels['Operating Profit'], latest_year, previous_year, default=0, scaling_factor=income_units)
        cash = get_value(balance_df, found_labels['Cash and Cash Equivalents'], latest_year, previous_year, default=0, scaling_factor=balance_units)
        revenue = get_value(income_df, found_labels['Revenue'], latest_year, previous_year, default=0, scaling_factor=income_units)

        try:
            tax_rate = get_value(ratios_df, found_labels['Effective Tax Rate'], latest_year, previous_year, default=None, scaling_factor=1)
            if pd.isna(tax_rate):
                raise ValueError("Effective Tax Rate is NaN")
            print(f"Effective Tax Rate from Financial Ratios: {tax_rate:.2%}")
        except (KeyError, ValueError):
            print(f"Warning: Could not use 'Effective Tax Rate' from Financial Ratios for {ticker}.")
            try:
                print(f"Income Before Tax: {income_before_tax}, Income Tax Expense: {income_tax_expense}")
                if pd.notna(income_before_tax) and pd.notna(income_tax_expense) and income_before_tax != 0:
                    tax_rate = income_tax_expense / income_before_tax
                    print(f"Calculated tax rate: {tax_rate:.2%}")
                else:
                    print(f"Warning: Invalid data for tax rate calculation. Using default tax rate of 21.0%.")
                    tax_rate = 0.21
            except (KeyError, TypeError) as e:
                print(f"Warning: Could not calculate tax rate from Income Statement: {e}. Using default tax rate of 21.0%.")
                tax_rate = 0.21

        roic = None
        if use_enhanced_model:
            invested_capital = total_equity + total_debt - cash if pd.notna(total_equity) and pd.notna(total_debt) and pd.notna(cash) else 1e9
            nopat = operating_profit * (1 - tax_rate) if pd.notna(operating_profit) and pd.notna(tax_rate) else 0
            roic = nopat / invested_capital if invested_capital != 0 and pd.notna(nopat) else 0.15
            print(f"Calculated ROIC: {roic:.2%}")

        cost_of_debt_pretax = net_non_operating_interest / total_debt if total_debt != 0 else 0
        cost_of_debt = cost_of_debt_pretax * (1 - tax_rate) if cost_of_debt_pretax != 0 else 0.05

        # Dynamic parameters based on ticker
        risk_free_rate = 0.03
        market_return = 0.09
        beta = 1.0
        terminal_growth_rate = 0.025
        growth_cap = 0.08  # Default for mature companies
        reinvestment_cap = 0.05  # Default for mature companies

        if ticker.endswith(".NS"):  # Indian stocks
            risk_free_rate = 0.055
            market_return = 0.105
            beta = 0.7
            terminal_growth_rate = 0.04
            growth_cap = 0.10
            reinvestment_cap = 0.05
        elif ticker == "AMZN":
            beta = 1.2
            terminal_growth_rate = 0.035
            growth_cap = 0.12
            reinvestment_cap = 0.10
        elif ticker == "NKE":
            beta = 0.9
            terminal_growth_rate = 0.025
            growth_cap = 0.08
            reinvestment_cap = 0.05

        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        total_capital = total_debt + total_equity if pd.notna(total_debt) and pd.notna(total_equity) else total_equity
        debt_weight = total_debt / total_capital if total_capital != 0 else 0
        equity_weight = total_equity / total_capital if total_capital != 0 else 1
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt)
        wacc = min(max(wacc, 0.05), 0.10) if pd.notna(wacc) else 0.08

        revenue_years = [y for y in all_years if int(y) <= start_year]
        revenue_years.sort()
        if len(revenue_years) >= 2 and found_labels['Revenue']:
            revenue_end = get_value(income_df, found_labels['Revenue'], revenue_years[-1], None, default=0, scaling_factor=income_units)
            revenue_start = get_value(income_df, found_labels['Revenue'], revenue_years[0], None, default=0, scaling_factor=income_units)
            years_span = int(revenue_years[-1]) - int(revenue_years[0])
            cagr = (revenue_end / revenue_start) ** (1 / years_span) - 1 if pd.notna(revenue_start) and revenue_start != 0 and years_span > 0 else 0.05
            print(f"Revenue CAGR from {revenue_years[0]} to {revenue_years[-1]}: {cagr:.2%}")
        else:
            cagr = 0.05
        high_growth_rate = min(max(cagr, 0.05), growth_cap) if pd.notna(cagr) else 0.05

        currency = 'INR' if ticker.endswith('.NS') else 'USD'

        inputs = {
            'starting_fcf': starting_fcf,
            'wacc': wacc,
            'high_growth_rate': high_growth_rate,
            'terminal_growth_rate': terminal_growth_rate,
            'net_debt': net_debt,
            'shares_outstanding': shares_outstanding,
            'current_price': current_price,
            'currency': currency,
            'roic': roic,
            'reinvestment_cap': reinvestment_cap
        }
        print("DCF Inputs:", {k: v for k, v in inputs.items()})
        for key, value in inputs.items():
            if key in ['starting_fcf', 'wacc', 'shares_outstanding'] and pd.isna(value):
                print(f"Critical input {key} is NaN")
                return None

        return inputs
    
    except Exception as e:
        print(f"Error processing data for {ticker}: {e}")
        return None

def run_dcf_monte_carlo(ticker, data_dir="data"):
    inputs = extract_dcf_inputs(ticker, data_dir)
    if inputs is None:
        print("Failed to extract inputs. Exiting.")
        return
    
    print("\nRunning Simple Model...")
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
        wacc_range=(0.05, 0.12),
        growth_range=(0.05, 0.12),
        terminal_growth_range=(0.02, 0.04),
        fcf_error=0.1
    )
    plot_simulation_results(
        simulated_values=simple_results, 
        current_price=inputs['current_price'], 
        ticker=ticker + "_Simple", 
        currency=inputs['currency']
    )

    print("\nRunning Enhanced Model...")
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
        wacc_range=(0.05, 0.12),
        growth_range=(0.05, 0.12),
        terminal_growth_range=(0.02, 0.04),
        fcf_error=0.1
    )
    plot_simulation_results(
        simulated_values=enhanced_results, 
        current_price=inputs['current_price'], 
        ticker=ticker + "_Enhanced", 
        currency=inputs['currency']
    )

if __name__ == "__main__":
    ticker = "BAJFINANCE.NS"  # Change this to any ticker
    run_dcf_monte_carlo(ticker)

