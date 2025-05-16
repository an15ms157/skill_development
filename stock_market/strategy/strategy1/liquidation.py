import pandas as pd
import re
import matplotlib.pyplot as plt
import os

ticker = "VBL.NS"
source = f"/home/abhishek/Desktop/skill_development/stock_market/valuation/data/{ticker}/"
balance_sheet_file = source + f"{ticker} Balance Sheet Statement (Annual) - Discounting Cash Flows.xlsx"
income_statement_file = source + f"{ticker} Income Statement (Annual) - Discounting Cash Flows.xlsx"

# Load the Balance Sheet
df = pd.read_excel(balance_sheet_file, header=None).transpose()
df.columns = df.iloc[0].str.strip()
df = df[1:].set_index(df.columns[0])
df.index = df.index.str.strip().str.replace('\n', '')

# Clean values
def clean_value(val):
    if pd.isna(val) or val == '':
        return 0.0
    if isinstance(val, str):
        val = re.sub(r'[^\d.-]', '', val)
        try:
            return float(val)
        except ValueError:
            return 0.0
    return float(val)

# Recovery rates
recovery_rates = {
    'Cash & Equivalents': 1.00,
    'Short Term Investments': 0.90,
    'Receivables': 0.60,
    'Inventory': 0.60,
    'Other Current Assets': 0.30,
    'Property, Plant and Equipment': 0.40,
    'Goodwill and Intangible Assets': 0.00,
    'Other Long Term Assets': 0.30
}

# Function to calculate liquidation value for any year
def get_liquidation_value_for_year(year):
    year = str(year)
    available_year_rows = [row for row in df.index if row.startswith(year)]
    if not available_year_rows:
        return None
    year_row = available_year_rows[0]
    print(f"Using year_row: {year_row}")
    df_year = df.loc[year_row]
    df_year = pd.DataFrame({'Account': df_year.index.str.strip(), 'Value': df_year.values})
    df_year['Value'] = df_year['Value'].apply(clean_value)
    asset_values = {account: (df_year[df_year['Account'] == account]['Value'].iloc[0] if account in df_year['Account'].values else 0.0) for account in recovery_rates}
    gross_liquidation_value = sum(value * recovery_rates[account] for account, value in asset_values.items())
    total_liabilities = df_year[df_year['Account'] == 'Total Liabilities']['Value'].iloc[0] if 'Total Liabilities' in df_year['Account'].values else 0.0
    liquidation_costs = gross_liquidation_value * 0.10
    net_liquidation_value = gross_liquidation_value - total_liabilities - liquidation_costs
    print(f"\nYear: {year_row}")
    print(f"  Gross Liquidation Value: ${gross_liquidation_value:,.2f}")
    print(f"  Total Liabilities: ${total_liabilities:,.2f}")
    print(f"  Liquidation Costs: ${liquidation_costs:,.2f}")
    print(f"  Net Liquidation Value: ${net_liquidation_value:,.2f}")
    return {
        'year_row': year_row,
        'net_liquidation_value': net_liquidation_value
    }

# Collect data for all years
years_to_check = [y for y in range(1950, 2026) if any(str(y) in idx for idx in df.index)]
data = [get_liquidation_value_for_year(year) for year in years_to_check if get_liquidation_value_for_year(year)]

# Extract years and net liquidation values
years = [int(str(d['year_row'])[:4]) for d in data]
net_liquidation_values = [d['net_liquidation_value'] for d in data]

# Get shares outstanding for each year
shares_data = {}
try:
    df_income = pd.read_excel(income_statement_file)
    df_income.columns = df_income.columns.str.strip().str.replace('\n', '')
    account_col = df_income.columns[0]
    print(f"\nIncome Statement columns: {df_income.columns.tolist()}")
    for year in years_to_check:
        year_str = str(year)
        year_cols = [col for col in df_income.columns if col.startswith(year_str)]
        if year_cols:
            year_col = sorted(year_cols)[-1]
            print(f"Checking year {year_str}, column: {year_col}")
            if 'Weighted Average Shares Outstanding' in df_income[account_col].values:
                raw_shares = df_income[df_income[account_col] == 'Weighted Average Shares Outstanding'][year_col].iloc[0]
                print(f"Raw Shares for {year}: {raw_shares}")
                shares = clean_value(raw_shares) * 1_000_000
                shares_data[year] = shares if shares > 0 else 1_000_000
            else:
                print(f"Warning: 'Weighted Average Shares Outstanding' not found for {year}")
                shares_data[year] = 1_000_000
        else:
            print(f"Warning: No {year_str} column found")
            shares_data[year] = 1_000_000
except Exception as e:
    print(f"Error reading shares data: {e}")
    for year in years_to_check:
        shares_data[year] = 1_000_000

# Calculate liquidation value per share
liquidation_value_per_share = []
for i, d in enumerate(data):
    year = int(str(d['year_row'])[:4])
    shares = shares_data.get(year, 1_000_000)
    nlps = d['net_liquidation_value'] / shares if shares > 0 else 0.0
    print(f"Year: {year}, Shares: {shares}, NLV: {d['net_liquidation_value']}, NLVPS: {nlps}")
    liquidation_value_per_share.append(nlps)

# Placeholder average prices (replace with actual data)
avg_prices = {year: 50 + (year - 1985) * 5 for year in years}
price_to_nlps_ratio = [avg_prices[year] / nlps if nlps != 0 else float('inf') for year, nlps in zip(years, liquidation_value_per_share)]

# Plotting
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax1.plot(years, net_liquidation_values, marker='o', label='Net Liquidation Value ($M)', color='blue')
ax1.set_xlabel('Year')
ax1.set_ylabel('Net Liquidation Value ($M)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title(f'{ticker} Net Liquidation Value Over Time')

ax2 = ax1.twinx()
# Remove Liquidation Value Per Share line
# ax2.plot(years, scaled_nlps, marker='s', label='Liquidation Value Per Share ($ x 10^6)', color='orange')
ax2.plot(years, price_to_nlps_ratio, marker='x', label='Price / Liquidation Value Per Share', color='green')
ax2.set_ylabel('Price / Liquidation Value Per Share (Ratio)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plot_dir = os.path.join('plots', 'liquidation')
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, f'{ticker}_net_liquidation_value_and_ratios.png'))
plt.show()

# Print results with higher precision
for year, nl_value, nlps, ratio in zip(years, net_liquidation_values, liquidation_value_per_share, price_to_nlps_ratio):
    print(f"Year: {year}")
    print(f"  Net Liquidation Value: ${nl_value:,.2f}")
    print(f"  Liquidation Value Per Share: ${nlps:.10f}")  # Increased precision
    print(f"  Price / Liquidation Value Per Share: {ratio:.2f}")