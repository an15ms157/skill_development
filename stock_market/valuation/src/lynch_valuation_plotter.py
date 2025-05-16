import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# ---------------- Configuration and Assumptions ----------------
ENABLE_LOGGING = True
TICKERS = ["COLPAL.NS"]
VALUATION_ASSUMPTIONS = {
    "cagr_period": 5,
    "min_cagr": 5.0,
    "default_dividend_yield": 0.0
}
EXCEL_PATH_TEMPLATE = "data/{ticker}/{ticker} Financial Ratios (Annual) - Discounting Cash Flows.xlsx"
METRICS = {
    "eps": "Earnings Per Share",
    "fcf": "Free Cash Flow Per Share",
    "dividend_yield": "Annual Dividend Yield",
    "book_value": "Book Value Per Share",
    "roe": "Return on Equity"
}
PLOT_CONFIG = {
    "figsize": (12, 8),
    "sma_period": 200,
    "cagr_ylim": (0, 100),
    "bar_width": 150
}

# ---------------- Utility Functions ----------------
def log(message):
    if ENABLE_LOGGING:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def calculate_growth_rate(start_value, end_value, years):
    if start_value <= 0 or end_value <= 0:
        log(f"Invalid growth rate: start={start_value}, end={end_value}")
        return None
    try:
        return ((end_value / start_value) ** (1 / years) - 1) * 100
    except Exception as e:
        log(f"Error calculating CAGR: {e}")
        return None

def clean_dataframe(df):
    df.index = df.index.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
    df.columns = ['2025' if col == 'LTM' else col.split(' ')[0].split('-')[0] for col in df.columns]
    return df

def load_and_prepare_data(ticker):
    file_path = EXCEL_PATH_TEMPLATE.format(ticker=ticker)
    try:
        df = pd.read_excel(file_path, sheet_name=0, header=0, index_col=0)
        log(f"Loaded Excel for {ticker}. Shape: {df.shape}")
        df = clean_dataframe(df)
        df_t = df.T
        df_t['Year'] = df_t.index.map(lambda x: '2025' if x == 'LTM' else x)
        df_t['Year'] = pd.to_numeric(df_t['Year'], errors='coerce').astype(int)
        df_t = df_t.dropna(subset=['Year'])

        for metric in METRICS.values():
            if metric in df_t.columns:
                df_t[metric] = pd.to_numeric(df_t[metric].astype(str).str.replace('%', ''), errors='coerce')

        return df_t
    except Exception as e:
        log(f"Error loading data for {ticker}: {e}")
        return None

def calculate_valuations(df, ticker):
    n_years = VALUATION_ASSUMPTIONS["cagr_period"]
    lynch_prices, lynch_adj_prices, lynch_fcf_values, fair_prices, valid_years, eps_cagrs = [], [], [], [], [], []

    has_dividend = METRICS["dividend_yield"] in df.columns
    can_calc_financial = METRICS["book_value"] in df.columns and METRICS["roe"] in df.columns

    for i in range(len(df) - n_years):
        year = df['Year'].iloc[i]
        eps_now = df[METRICS["eps"]].iloc[i] if METRICS["eps"] in df.columns else None
        eps_past = df[METRICS["eps"]].iloc[i + n_years] if METRICS["eps"] in df.columns else None
        fcf_now = df[METRICS["fcf"]].iloc[i] if METRICS["fcf"] in df.columns else None
        fcf_past = df[METRICS["fcf"]].iloc[i + n_years] if METRICS["fcf"] in df.columns else None
        dividend_yield = df[METRICS["dividend_yield"]].iloc[i] / 100 if has_dividend and not pd.isna(df[METRICS["dividend_yield"]].iloc[i]) else VALUATION_ASSUMPTIONS["default_dividend_yield"]

        eps_valid = pd.notna(eps_now) and pd.notna(eps_past) and eps_now > 0 and eps_past > 0
        fcf_valid = pd.notna(fcf_now) and pd.notna(fcf_past) and fcf_now > 0 and fcf_past > 0

        eps_cagr = calculate_growth_rate(eps_past, eps_now, n_years) if eps_valid else None
        fcf_cagr = calculate_growth_rate(fcf_past, fcf_now, n_years) if fcf_valid else None

        if eps_cagr is not None and eps_cagr >= 0 and eps_valid:
            lynch_price = eps_now * eps_cagr
            lynch_adj_price = eps_now * (eps_cagr + dividend_yield * 100)
            eps_cagrs.append(eps_cagr)
        elif fcf_cagr is not None and fcf_cagr >= 0 and fcf_valid:
            lynch_price = fcf_now * fcf_cagr
            lynch_adj_price = fcf_now * (fcf_cagr + dividend_yield * 100)
            eps_cagrs.append(fcf_cagr)
        else:
            log(f"Skipping {ticker} {year} - No valid EPS or FCF CAGR")
            continue

        lynch_fcf_value = fcf_now * (fcf_cagr + dividend_yield * 100) if fcf_valid and fcf_cagr is not None else None

        fair_price = None
        if can_calc_financial:
            book = df[METRICS["book_value"]].iloc[i]
            roe = df[METRICS["roe"]].iloc[i]
            if pd.notna(book) and pd.notna(roe) and book > 0 and roe > 0:
                fair_price = book * (roe + dividend_yield * 100)
            elif fcf_valid and fcf_cagr is not None:
                fair_price = fcf_now * fcf_cagr  # fallback using FCF CAGR

        lynch_prices.append(lynch_price)
        lynch_adj_prices.append(lynch_adj_price)
        lynch_fcf_values.append(lynch_fcf_value if lynch_fcf_value is not None else 0)
        fair_prices.append(fair_price if fair_price is not None else 0)
        valid_years.append(year)

    if not valid_years:
        log(f"No valid valuation data for {ticker}")
        return None

    return {
        "years": valid_years[::-1],
        "lynch_prices": lynch_prices[::-1],
        "lynch_adj_prices": lynch_adj_prices[::-1],
        "lynch_fcf_values": lynch_fcf_values[::-1],
        "fair_prices": fair_prices[::-1],
        "eps_cagrs": eps_cagrs[::-1]
    }

def fetch_price_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        price_data = stock.history(period="max")['Close']
        price_data_weekly = price_data.resample('W-FRI').last().dropna()
        sma = price_data_weekly.rolling(window=PLOT_CONFIG["sma_period"]).mean() if len(price_data_weekly) >= PLOT_CONFIG["sma_period"] else None
        return price_data_weekly, sma
    except Exception as e:
        log(f"Error fetching price data for {ticker}: {e}")
        return None, None

def plot_valuations(ticker, valuations):
    n_years = VALUATION_ASSUMPTIONS["cagr_period"]
    if not valuations:
        log(f"Skipping plot for {ticker}: No valuation data")
        return

    fig, ax1 = plt.subplots(figsize=PLOT_CONFIG["figsize"])
    lynch_dates = pd.to_datetime([f"{year}-01-01" for year in valuations["years"]])
    price_data_weekly, sma = fetch_price_data(ticker)
    if price_data_weekly is None:
        log(f"Skipping plot for {ticker}: No price data")
        return

    ax1.plot(price_data_weekly.index, price_data_weekly, label=f'{ticker} Weekly Price', color='blue')
    if sma is not None:
        ax1.plot(sma.index, sma, label=f'{PLOT_CONFIG["sma_period"]}-Week SMA', color='red', linestyle='--')

    ax1.plot(lynch_dates, valuations["lynch_prices"], label=f'Lynch Valuation ({n_years}-yr)', marker='o', color='green')
    ax1.plot(lynch_dates, valuations["lynch_adj_prices"], label=f'Lynch Adj Valuation ({n_years}-yr + Div)', marker='s', color='orange')
    ax1.plot(lynch_dates, valuations["lynch_fcf_values"], label=f'Lynch FCF Valuation', marker='^', color='purple')
    if any(valuations["fair_prices"]):
        ax1.plot(lynch_dates, valuations["fair_prices"], label='Fair Price (Financial/FCF)', marker='d', color='red')

    ax1.set_title(f'{ticker} Weekly Price and Valuations ({n_years}-Year CAGR)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Price (Log Scale)')
    ax1.set_yscale('log')
    ax1.grid(True, which='both')
    ax1.legend(loc='upper left')
    ax1.set_xticks(lynch_dates)
    ax1.set_xticklabels(valuations["years"], rotation=45)

    formula_text = (
        f"Lynch = EPS × CAGR or FCF × FCF CAGR\n"
        f"Adj = EPS × (CAGR + Div) or FCF × (FCF CAGR + Div)\n"
        f"Fair Price = Book × (ROE + Div) or fallback to FCF × FCF CAGR"
    )
    ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2 = ax1.twinx()
    ax2.bar(lynch_dates, valuations["eps_cagrs"], alpha=0.3, color='green', label='CAGR (%)', width=PLOT_CONFIG["bar_width"])
    ax2.set_ylabel('CAGR (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(*PLOT_CONFIG["cagr_ylim"])
    ax2.legend(loc='upper right')

    filename = f'{ticker}_{n_years}yr_weekly_lynch_cagr_formula_fcf_fairprice.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    log(f"Saved plot: {filename}")

def main():
    for ticker in TICKERS:
        log(f"\nProcessing {ticker} with {VALUATION_ASSUMPTIONS['cagr_period']}-year CAGR")
        df = load_and_prepare_data(ticker)
        if df is None:
            continue
        valuations = calculate_valuations(df, ticker)
        if valuations is None:
            continue
        plot_valuations(ticker, valuations)

if __name__ == "__main__":
    main()
    log("Processing complete.")
