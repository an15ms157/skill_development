import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from config import logger

# Configuration: Data-Driven Parameters
CONFIG = {
    # General Settings
    'ticker': 'JPM',  # Stock ticker symbol
    'data_dir': '../data',  # Directory containing financial data Excel files
    'start_year': 1990,  # Start year for historical data (5 years back)
    'end_year': 2024,  # Latest year for data
    'currency': 'USD',  # Currency (derived from ticker)

    # Financial Data Defaults (used only if data is missing)
    'default_fcf': 1000e6,  # Default Free Cash Flow (1 billion USD)
    'default_net_debt': 0,  # Default Net Debt
    'default_shares_outstanding': 100e6,  # Default Shares Outstanding (100 million)
    'default_price_per_share': 100,  # Default Price Per Share
    'default_total_debt': 0,  # Default Total Debt
    'default_total_equity': 1000e6,  # Default Total Equity (1 billion)
    'default_tax_rate': 0.21,  # Default Effective Tax Rate (21%)
    'default_operating_profit': 0,  # Default Operating Profit
    'default_cash': 0,  # Default Cash and Cash Equivalents
    'default_revenue': 0,  # Default Revenue
    'default_nopat': 0,  # Default NOPAT
    'default_roic': 0.10,  # Default ROIC (10% for financial firms)
    'default_capex': 0,  # Default Capital Expenditures
    'default_working_capital_change': 0,  # Default Change in Working Capital
    'default_beta': 1.0,  # Default Beta

    # Market Data (updated as of May 2025)
    'risk_free_rate': 0.045,  # 10-year Treasury yield (4.5%)
    'market_return': 0.09,  # Historical 10-year S&P 500 return (9%)
    'gdp_growth': 0.02,  # Long-term U.S. GDP growth (2%)

    # Model Settings
    'high_growth_years': 10,  # Number of high-growth years
    'growth_cap': 0.10,  # Max high-growth rate (10%)
    'reinvestment_cap': 0.05,  # Max reinvestment rate (5%)
    'wacc_min': 0.05,  # Minimum WACC (5%)
    'wacc_max': 0.10,  # Maximum WACC (10%)

    # Monte Carlo Simulation Settings
    'num_simulations': 10000,  # Number of simulations
    'fcf_error': 0.1,  # Default FCF volatility (±10%)
    'net_debt_error': 0.05,  # Default Net debt volatility (±5%)
}

# DCF Model Class
class DCFModel:
    def __init__(self, starting_fcf, wacc, high_growth_rate, terminal_growth_rate, net_debt, shares_outstanding,
                 high_growth_years, growth_decay, margin_compression, roic, wacc_drift_std, use_enhanced_model,
                 ticker, reinvestment_cap):
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
        if not pd.notna(self.wacc) or self.wacc <= self.terminal_growth_rate + 0.02:
            raise ValueError(f"Invalid WACC: {self.wacc} (must be > terminal growth rate + 2%: {self.terminal_growth_rate + 0.02})")
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
                    reinvestment_rate = (growth / self.roic) * 0.3  # Data-driven multiplier
                    reinvestment_rate = min(max(reinvestment_rate, 0), self.reinvestment_cap)
                    fcf *= (1 - reinvestment_rate)
                growth = max(growth - self.growth_decay, 0.03)
                fcf *= (1 + growth)
                wacc += np.random.normal(0, self.wacc_drift_std)
                wacc = min(max(wacc, CONFIG['wacc_min']), CONFIG['wacc_max'])
            else:
                fcf *= (1 + self.high_growth_rate)
            fcfs.append(fcf)
        return fcfs
    
    def calculate_terminal_value(self, final_fcf):
        return final_fcf * (1 + self.terminal_growth_rate) / (self.wacc - self.terminal_growth_rate)
    
    def calculate_dcf(self):
        fcfs = self.project_fcfs()
        terminal_value = self.calculate_terminal_value(fcfs[-1])
        
        pv_fcfs = sum(fcf / (1 + self.wacc) ** (i + 1) for i, fcf in enumerate(fcfs))
        pv_terminal = terminal_value / (1 + self.wacc) ** self.high_growth_years
        
        enterprise_value = pv_fcfs + pv_terminal
        equity_value = max(enterprise_value - self.net_debt, 0)
        fair_value_per_share = equity_value / self.shares_outstanding if equity_value > 0 else 0
        
        return fair_value_per_share

# Monte Carlo Simulation
def run_monte_carlo(base_model, num_simulations, wacc_std, high_growth_std, terminal_growth_std, fcf_error, net_debt_error):
    results = []
    failed_simulations = 0
    for i in range(num_simulations):
        try:
            wacc = np.random.normal(base_model.wacc, wacc_std)
            wacc = max(wacc, CONFIG['wacc_min'])
            high_growth = np.random.normal(base_model.high_growth_rate, high_growth_std)
            terminal_growth = np.random.normal(base_model.terminal_growth_rate, terminal_growth_std)
            terminal_growth = min(terminal_growth, wacc - 0.02)
            fcf_random = np.random.normal(base_model.starting_fcf, base_model.starting_fcf * fcf_error)
            net_debt_adjustment = np.random.normal(0, abs(base_model.net_debt) * net_debt_error)
            net_debt = base_model.net_debt + net_debt_adjustment

            model = DCFModel(
                starting_fcf=max(fcf_random, 0.01),
                wacc=wacc,
                high_growth_rate=high_growth,
                terminal_growth_rate=terminal_growth,
                net_debt=net_debt,
                shares_outstanding=base_model.shares_outstanding,
                high_growth_years=base_model.high_growth_years,
                growth_decay=base_model.growth_decay,
                margin_compression=base_model.margin_compression,
                roic=base_model.roic,
                wacc_drift_std=base_model.wacc_drift_std,
                use_enhanced_model=base_model.use_enhanced_model,
                ticker=base_model.ticker,
                reinvestment_cap=base_model.reinvestment_cap
            )
            fair_value = model.calculate_dcf()
            if pd.notna(fair_value) and fair_value >= 0:
                results.append(fair_value)
        except Exception as e:
            failed_simulations += 1
            if i < 10:
                print(f"Simulation {i} failed: {e}")
    
    if failed_simulations > 0:
        print(f"Warning: {failed_simulations} out of {num_simulations} simulations failed.")
    
    return np.array(results)

# Plot Simulation Results with Sensitivity Analysis
def plot_simulation_results(simulated_values, current_price, ticker, currency, model_type, inputs):
    if len(simulated_values) == 0:
        print(f"Error: No valid simulation results for {ticker} ({model_type}). Check input data.")
        return
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.histplot(simulated_values, bins=100, kde=True, color='skyblue')
    
    mean_val = np.mean(simulated_values)
    std_dev = np.std(simulated_values)
    
    plt.axvline(mean_val, color='green', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_dev, color='orange', linestyle='--', label=f'+1σ: {mean_val + std_dev:.2f}')
    plt.axvline(mean_val - std_dev, color='orange', linestyle='--', label=f'-1σ: {mean_val - std_dev:.2f}')
    
    if current_price and pd.notna(current_price):
        plt.axvline(current_price, color='black', linestyle='-', label=f'Current Price: {current_price:.2f}')
    
    plt.title(f'DCF Monte Carlo Simulation Results for {ticker} ({model_type})')
    plt.xlabel(f'Fair Value per Share ({currency})')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{ticker}_{model_type}_dcf_monte_carlo.png")
    plt.show()
    
    print(f"\n{model_type} Model Results for {ticker}:")
    print(f"Mean Fair Value: {mean_val:.2f} {currency}")
    print(f"1σ Range: {mean_val - std_dev:.2f} to {mean_val + std_dev:.2f}")
    
    # Sensitivity Analysis
    print("\nSensitivity Analysis:")
    for param, change in [('WACC', 0.01), ('High Growth Rate', 0.01), ('Terminal Growth Rate', 0.005)]:
        temp_model = DCFModel(
            starting_fcf=inputs['starting_fcf'],
            wacc=inputs['wacc'] + change if param == 'WACC' else inputs['wacc'],
            high_growth_rate=inputs['high_growth_rate'] + change if param == 'High Growth Rate' else inputs['high_growth_rate'],
            terminal_growth_rate=inputs['terminal_growth_rate'] + change if param == 'Terminal Growth Rate' else inputs['terminal_growth_rate'],
            net_debt=inputs['net_debt'],
            shares_outstanding=inputs['shares_outstanding'],
            high_growth_years=CONFIG['high_growth_years'],
            growth_decay=inputs.get('growth_decay', 0),
            margin_compression=inputs.get('margin_compression', 0),
            roic=inputs['roic'],
            wacc_drift_std=inputs.get('wacc_drift_std', 0),
            use_enhanced_model=model_type == 'Enhanced',
            ticker=ticker,
            reinvestment_cap=inputs['reinvestment_cap']
        )
        new_value = temp_model.calculate_dcf()
        print(f"{param} +{change*100:.1f}%: Fair Value = {new_value:.2f} ({(new_value - mean_val)/mean_val*100:.1f}% change)")

# Unit Detection for Financial Data
def detect_units(df, ticker, context):
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
    print(f"No unit indicators found for {ticker} {context}. Assuming units in millions.")
    return 1e6

# Extract DCF Inputs from Financial Data
def extract_dcf_inputs(ticker, data_dir, start_year, end_year):
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

        financial_scaling_factor = max([income_units, balance_units, cashflow_units])
        print(f"Using consistent financial scaling factor for {ticker}: {financial_scaling_factor}")

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
            'Interest Expense': ['Interest Expense', 'Net Interest Expense'],
            'Operating Profit': ['Operating Profit', 'Operating Income', 'EBIT'],
            'Cash and Cash Equivalents': ['Cash and Cash Equivalents', 'Cash & Equivalents', 'Cash', 'Cash and Short Term Investments'],
            'Revenue': ['Revenue', 'Total Revenue'],
            'Depreciation': ['Depreciation', 'Depreciation and Amortization'],
            'Working Capital Change': ['Change in Working Capital', 'Changes in Working Capital'],
            'Capital Expenditures': ['Capital Expenditures', 'CapEx', 'Purchases of Property, Plant, and Equipment'],
            'Beta': ['Beta']
        }
        row_to_df = {
            'Free Cash Flow': cashflow_df,
            'Net Debt': cashflow_df,
            'Weighted Average Shares Outstanding': income_df,
            'Price Per Share': ratios_df,
            'Total Debt': balance_df,
            'Total Equity': balance_df,
            'Income Before Tax': income_df,
            'Income Tax Expense': income_df,
            'Interest Expense': income_df,
            'Operating Profit': income_df,
            'Cash and Cash Equivalents': balance_df,
            'Revenue': income_df,
            'Depreciation': cashflow_df,
            'Working Capital Change': cashflow_df,
            'Capital Expenditures': cashflow_df,
            'Beta': ratios_df
        }

        found_labels = {}
        for key, targets in row_labels.items():
            found_labels[key] = find_row_label(row_to_df[key], targets)
            if not found_labels[key] and key in ['Free Cash Flow', 'Weighted Average Shares Outstanding', 'Revenue']:
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

        def convert_to_numeric(value, is_shares=False, is_price=False, is_beta=False, scaling_factor=1):
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
                return float(value) * (scaling_factor if not is_beta else 1)
            return np.nan

        def get_value(df, row, year, prev_year, default, is_shares=False, is_price=False, is_beta=False, scaling_factor=1):
            if row not in df.index:
                print(f"Warning: {row} not in DataFrame index. Using default: {default}")
                return default
            raw_value = df.loc[row, year]
            print(f"Raw {row} for {year}: {raw_value}")
            value = convert_to_numeric(raw_value, is_shares, is_price, is_beta, scaling_factor)
            print(f"Converted {row} for {year} with scaling factor {scaling_factor}: {value}")
            if pd.notna(value):
                return value
            if prev_year and prev_year in df.columns:
                raw_value = df.loc[row, prev_year]
                print(f"Raw {row} for {prev_year}: {raw_value}")
                value = convert_to_numeric(raw_value, is_shares, is_price, is_beta, scaling_factor)
                print(f"Converted {row} for {prev_year} with scaling factor {scaling_factor}: {value}")
                if pd.notna(value):
                    print(f"Using {row} from {prev_year}: {value}")
                    return value
            print(f"Warning: {row} not found in {year} or {prev_year}. Using default: {default}")
            return default

        # Financial Inputs (ordered to ensure dependencies are resolved)
        operating_profit = get_value(income_df, found_labels['Operating Profit'], latest_year, previous_year,
                                    CONFIG['default_operating_profit'], scaling_factor=financial_scaling_factor)
        income_before_tax = get_value(income_df, found_labels['Income Before Tax'], latest_year, previous_year,
                                     1, scaling_factor=financial_scaling_factor)
        income_tax_expense = get_value(income_df, found_labels['Income Tax Expense'], latest_year, previous_year,
                                      0, scaling_factor=financial_scaling_factor)
        depreciation = get_value(cashflow_df, found_labels['Depreciation'], latest_year, previous_year,
                                CONFIG['default_capex'], scaling_factor=financial_scaling_factor)
        capex = get_value(cashflow_df, found_labels['Capital Expenditures'], latest_year, previous_year,
                         CONFIG['default_capex'], scaling_factor=financial_scaling_factor)
        working_capital_change = get_value(cashflow_df, found_labels['Working Capital Change'], latest_year, previous_year,
                                          CONFIG['default_working_capital_change'], scaling_factor=financial_scaling_factor)

        # Starting FCF: Median of positive historical FCFs or estimated
        historical_fcfs = []
        for year in available_years:
            if found_labels['Free Cash Flow'] in cashflow_df.index and year in cashflow_df.columns:
                fcf = convert_to_numeric(cashflow_df.loc[found_labels['Free Cash Flow'], year], scaling_factor=financial_scaling_factor)
                if pd.notna(fcf) and fcf > 0:
                    historical_fcfs.append(fcf)
        if historical_fcfs:
            starting_fcf = np.median(historical_fcfs)
            print(f"Starting FCF: Median of positive historical FCFs {historical_fcfs} = {starting_fcf:.2f}")
        else:
            # Estimate FCF: NOPAT + Depreciation - CapEx - ΔWorking Capital
            tax_rate = income_tax_expense / income_before_tax if pd.notna(income_before_tax) and income_before_tax != 0 else CONFIG['default_tax_rate']
            nopat = operating_profit * (1 - tax_rate)
            starting_fcf = nopat + depreciation - capex - working_capital_change
            print(f"Starting FCF: Estimated as NOPAT ({nopat:.2f}) + Depreciation ({depreciation:.2f}) - CapEx ({capex:.2f}) - ΔWC ({working_capital_change:.2f}) = {starting_fcf:.2f}")
        if starting_fcf <= 0:
            starting_fcf = CONFIG['default_fcf']
            print(f"Warning: Estimated FCF non-positive. Using default: {starting_fcf}")

        # Other Financial Inputs
        net_debt = get_value(cashflow_df, found_labels['Net Debt'], latest_year, previous_year,
                            CONFIG['default_net_debt'], scaling_factor=financial_scaling_factor)
        shares_outstanding = get_value(income_df, found_labels['Weighted Average Shares Outstanding'], latest_year,
                                      previous_year, CONFIG['default_shares_outstanding'], is_shares=True,
                                      scaling_factor=detect_units(income_df, ticker, "Shares Outstanding"))
        current_price = get_value(ratios_df, found_labels['Price Per Share'], latest_year, previous_year,
                                 CONFIG['default_price_per_share'], is_price=True, scaling_factor=1)
        total_debt = get_value(balance_df, found_labels['Total Debt'], latest_year, previous_year,
                              CONFIG['default_total_debt'], scaling_factor=financial_scaling_factor)
        total_equity = get_value(balance_df, found_labels['Total Equity'], latest_year, previous_year,
                                CONFIG['default_total_equity'], scaling_factor=financial_scaling_factor)
        interest_expense = get_value(income_df, found_labels['Interest Expense'], latest_year, previous_year,
                                   0, scaling_factor=financial_scaling_factor)
        cash = get_value(balance_df, found_labels['Cash and Cash Equivalents'], latest_year, previous_year,
                        CONFIG['default_cash'], scaling_factor=financial_scaling_factor)
        revenue = get_value(income_df, found_labels['Revenue'], latest_year, previous_year,
                           CONFIG['default_revenue'], scaling_factor=financial_scaling_factor)

        # Tax Rate
        tax_rate = income_tax_expense / income_before_tax if pd.notna(income_before_tax) and income_before_tax != 0 else CONFIG['default_tax_rate']
        print(f"Tax Rate: {tax_rate:.2%} (Income Tax Expense: {income_tax_expense}, Income Before Tax: {income_before_tax})")

        # ROIC
        invested_capital = total_equity + total_debt - cash if pd.notna(total_equity) and pd.notna(total_debt) and pd.notna(cash) else 1e9
        nopat = operating_profit * (1 - tax_rate) if pd.notna(operating_profit) else CONFIG['default_nopat']
        roic = nopat / invested_capital if invested_capital > 0 and pd.notna(nopat) else CONFIG['default_roic']
        print(f"ROIC: {roic:.2%} (NOPAT: {nopat:.2f}, Invested Capital: {invested_capital:.2f})")

        # WACC
        beta = get_value(ratios_df, found_labels['Beta'], latest_year, previous_year, CONFIG['default_beta'], is_beta=True, scaling_factor=1)
        cost_of_equity = CONFIG['risk_free_rate'] + beta * (CONFIG['market_return'] - CONFIG['risk_free_rate'])
        cost_of_debt = (interest_expense / total_debt * (1 - tax_rate)) if total_debt != 0 and pd.notna(interest_expense) and interest_expense != 0 else 0.05
        total_capital = total_debt + total_equity if pd.notna(total_debt) and pd.notna(total_equity) else total_equity
        debt_weight = total_debt / total_capital if total_capital != 0 else 0
        equity_weight = total_equity / total_capital if total_capital != 0 else 1
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt)
        wacc = min(max(wacc, CONFIG['wacc_min']), CONFIG['wacc_max'])
        print(f"WACC: {wacc:.2%} (Cost of Equity: {cost_of_equity:.2%}, Cost of Debt: {cost_of_debt:.2%}, Equity Weight: {equity_weight:.2f}, Debt Weight: {debt_weight:.2f})")

        # High-Growth Rate: Revenue CAGR
        revenue_years = [y for y in all_years if start_year <= int(y) <= int(latest_year)]
        revenue_years.sort()
        revenues = []
        for year in revenue_years:
            rev = get_value(income_df, found_labels['Revenue'], year, None, CONFIG['default_revenue'], scaling_factor=financial_scaling_factor)
            if pd.notna(rev):
                revenues.append((year, rev))
        if len(revenues) >= 2:
            revenue_end = revenues[-1][1]
            revenue_start = revenues[0][1]
            years_span = int(revenues[-1][0]) - int(revenues[0][0])
            cagr = (revenue_end / revenue_start) ** (1 / years_span) - 1 if revenue_start != 0 and years_span > 0 else 0.05
            print(f"Revenue CAGR from {revenues[0][0]} to {revenues[-1][0]}: {cagr:.2%} (Start: {revenue_start:.2f}, End: {revenue_end:.2f})")
        else:
            cagr = 0.05
            print(f"Insufficient revenue data. Using default CAGR: {cagr:.2%}")
        high_growth_rate = min(max(cagr, 0.03), CONFIG['growth_cap'])

        # Terminal Growth Rate: Average of GDP growth and risk-free rate
        terminal_growth_rate = min((CONFIG['gdp_growth'] + CONFIG['risk_free_rate']) / 2, wacc - 0.02)
        print(f"Terminal Growth Rate: {terminal_growth_rate:.2%} (Average of GDP Growth: {CONFIG['gdp_growth']:.2%}, Risk-Free Rate: {CONFIG['risk_free_rate']:.2%})")

        # Monte Carlo Parameters: Data-driven volatility
        fcf_volatility = np.std(historical_fcfs) / np.mean(historical_fcfs) if historical_fcfs else CONFIG['fcf_error']
        wacc_std = 0.01  # ±1% based on historical WACC variability
        high_growth_std = cagr / 5 if cagr > 0 else 0.01  # ±20% of CAGR
        terminal_growth_std = 0.005  # ±0.5%
        growth_decay = cagr / 50 if cagr > 0 else 0.0002  # 2% of CAGR
        margin_compression = 0.0005  # Based on historical margin trends
        wacc_drift_std = 0.002  # ±0.2% per year
        print(f"Monte Carlo Parameters: FCF Volatility: {fcf_volatility:.2%}, WACC Std: {wacc_std:.2%}, High Growth Std: {high_growth_std:.2%}, Terminal Growth Std: {terminal_growth_std:.2%}")
        print(f"Enhanced Model Parameters: Growth Decay: {growth_decay:.4f}, Margin Compression: {margin_compression:.4f}, WACC Drift Std: {wacc_drift_std:.4f}")

        inputs = {
            'starting_fcf': starting_fcf,
            'wacc': wacc,
            'high_growth_rate': high_growth_rate,
            'terminal_growth_rate': terminal_growth_rate,
            'net_debt': net_debt,
            'shares_outstanding': shares_outstanding,
            'current_price': current_price,
            'currency': CONFIG['currency'],
            'roic': roic,
            'reinvestment_cap': CONFIG['reinvestment_cap'],
            'growth_decay': growth_decay,
            'margin_compression': margin_compression,
            'wacc_drift_std': wacc_drift_std,
            'fcf_error': fcf_volatility,
            'wacc_std': wacc_std,
            'high_growth_std': high_growth_std,
            'terminal_growth_std': terminal_growth_std
        }
        print("\nDCF Inputs Summary:")
        for k, v in inputs.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        logger.info(f"Successfully extracted DCF inputs for {ticker}.")
        return inputs
    
    except Exception as e:
        logger.error(f"Error extracting DCF inputs for {ticker}: {e}")
        return None

# Main Function to Run DCF Monte Carlo Simulation
def run_dcf_monte_carlo(ticker, data_dir, start_year, end_year):
    inputs = extract_dcf_inputs(ticker, data_dir, start_year, end_year)
    if inputs is None:
        print("Failed to extract inputs. Exiting.")
        return

    # Simple Model
    print("\nRunning Simple Model...")
    simple_model = DCFModel(
        starting_fcf=inputs['starting_fcf'],
        wacc=inputs['wacc'],
        high_growth_rate=inputs['high_growth_rate'],
        terminal_growth_rate=inputs['terminal_growth_rate'],
        net_debt=inputs['net_debt'],
        shares_outstanding=inputs['shares_outstanding'],
        high_growth_years=CONFIG['high_growth_years'],
        growth_decay=0,
        margin_compression=0,
        roic=inputs['roic'],
        wacc_drift_std=0,
        use_enhanced_model=False,
        ticker=ticker,
        reinvestment_cap=inputs['reinvestment_cap']
    )
    simple_results = run_monte_carlo(
        simple_model,
        num_simulations=CONFIG['num_simulations'],
        wacc_std=inputs['wacc_std'],
        high_growth_std=inputs['high_growth_std'],
        terminal_growth_std=inputs['terminal_growth_std'],
        fcf_error=inputs['fcf_error'],
        net_debt_error=CONFIG['net_debt_error']
    )
    plot_simulation_results(
        simulated_values=simple_results,
        current_price=inputs['current_price'],
        ticker=ticker,
        currency=inputs['currency'],
        model_type='Simple',
        inputs=inputs
    )

    # Enhanced Model
    print("\nRunning Enhanced Model...")
    enhanced_model = DCFModel(
        starting_fcf=inputs['starting_fcf'],
        wacc=inputs['wacc'],
        high_growth_rate=inputs['high_growth_rate'],
        terminal_growth_rate=inputs['terminal_growth_rate'],
        net_debt=inputs['net_debt'],
        shares_outstanding=inputs['shares_outstanding'],
        high_growth_years=CONFIG['high_growth_years'],
        growth_decay=inputs['growth_decay'],
        margin_compression=inputs['margin_compression'],
        roic=inputs['roic'],
        wacc_drift_std=inputs['wacc_drift_std'],
        use_enhanced_model=True,
        ticker=ticker,
        reinvestment_cap=inputs['reinvestment_cap']
    )
    enhanced_results = run_monte_carlo(
        enhanced_model,
        num_simulations=CONFIG['num_simulations'],
        wacc_std=inputs['wacc_std'],
        high_growth_std=inputs['high_growth_std'],
        terminal_growth_std=inputs['terminal_growth_std'],
        fcf_error=inputs['fcf_error'],
        net_debt_error=CONFIG['net_debt_error']
    )
    plot_simulation_results(
        simulated_values=enhanced_results,
        current_price=inputs['current_price'],
        ticker=ticker,
        currency=inputs['currency'],
        model_type='Enhanced',
        inputs=inputs
    )

def plot_operating_cash_flow(operating_cash_flows, years, ticker):
    """Plot Operating Cash Flow over the years."""
    plt.figure(figsize=(12, 6))
    plt.plot(years, operating_cash_flows, marker='o', linestyle='-', color='blue', label='Operating Cash Flow')
    plt.title(f'Operating Cash Flow Over the Years for {ticker}')
    plt.xlabel('Year')
    plt.ylabel('Operating Cash Flow (in millions)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{ticker}_operating_cash_flow.png")
    plt.show()

# Example usage (replace with actual data):
# years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
# operating_cash_flows = [500, 550, 600, 580, 620, 700, 750, 800, 850, 900]
# plot_operating_cash_flow(operating_cash_flows, years, 'JPM')

if __name__ == "__main__":
    run_dcf_monte_carlo(
        ticker=CONFIG['ticker'],
        data_dir=CONFIG['data_dir'],
        start_year=CONFIG['start_year'],
        end_year=CONFIG['end_year']
    )