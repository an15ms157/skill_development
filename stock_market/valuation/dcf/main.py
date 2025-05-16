from config import CONFIG, logger
from dcf_model import DCFModel
from monte_carlo import run_monte_carlo
from plotting import plot_simulation_results
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class FinancialAssumptions:
    revenue_growth_rate: float
    operating_margin: float
    tax_rate: float
    depreciation_pct: float
    capex_pct: float
    nwc_pct: float
    discount_rate: float
    perpetual_growth_rate: float

def project_cash_flows(
    revenue: float,
    assumptions: FinancialAssumptions,
    years: int = 5
) -> Dict[int, float]:
    """Projects free cash flows for a given number of years based on financial assumptions."""
    cash_flows = {}

    for year in range(1, years + 1):
        revenue *= 1 + assumptions.revenue_growth_rate
        operating_income = revenue * assumptions.operating_margin
        tax = operating_income * assumptions.tax_rate
        net_operating_profit = operating_income - tax

        depreciation = revenue * assumptions.depreciation_pct
        capex = revenue * assumptions.capex_pct
        nwc_investment = revenue * assumptions.nwc_pct

        fcff = net_operating_profit + depreciation - capex - nwc_investment
        cash_flows[year] = fcff

    return cash_flows

def calculate_terminal_value(
    final_year_cash_flow: float,
    assumptions: FinancialAssumptions
) -> float:
    """Calculates the terminal value based on the final year's cash flow and assumptions."""
    return (
        final_year_cash_flow * (1 + assumptions.perpetual_growth_rate)
    ) / (assumptions.discount_rate - assumptions.perpetual_growth_rate)

def calculate_dcf_value(
    initial_revenue: float,
    assumptions: FinancialAssumptions
) -> float:
    """Calculates the DCF value based on projected cash flows and terminal value."""
    cash_flows = project_cash_flows(initial_revenue, assumptions)

    dcf_value = 0.0
    for year, fcff in cash_flows.items():
        dcf_value += fcff / (1 + assumptions.discount_rate) ** year

    terminal_value = calculate_terminal_value(
        cash_flows[max(cash_flows.keys())], assumptions
    )
    dcf_value += terminal_value / (1 + assumptions.discount_rate) ** max(cash_flows.keys())

    return dcf_value

# Example usage
if __name__ == "__main__":
    logger.info("Starting DCF Monte Carlo simulation for JPM")

    # Log the current configuration and settings
    logger.info("Current Configuration and Settings:")
    for key, value in CONFIG.items():
        logger.info(f"{key}: {value}")

    # Inputs derived from Excel data (2024), adjusted for realism
    inputs = {
        'starting_fcf': CONFIG['default_fcf'],
        'wacc': CONFIG['default_wacc'],
        'high_growth_rate': 0.04,  # Base rate, scenarios adjust in dcf_model
        'terminal_growth_rate': CONFIG['gdp_growth'],
        'net_debt': CONFIG['default_net_debt'],
        'shares_outstanding': CONFIG['default_shares_outstanding'],
        'current_price': CONFIG['default_price_per_share'],
        'currency': CONFIG['currency'],
        'roic': CONFIG['default_roic'],
        'reinvestment_cap': CONFIG['reinvestment_cap'],
        'growth_decay': CONFIG['growth_decay'],
        'margin_compression': CONFIG['default_margin_compression'] if 'default_margin_compression' in CONFIG else 0.005,
        'wacc_drift_std': CONFIG['default_wacc_drift_std'] if 'default_wacc_drift_std' in CONFIG else 0.002,
        'fcf_error': CONFIG['fcf_error'],
        'wacc_std': CONFIG['wacc_std'] if 'wacc_std' in CONFIG else 0.005,
        'high_growth_std': CONFIG['high_growth_std'] if 'high_growth_std' in CONFIG else 0.005,
        'terminal_growth_std': CONFIG['terminal_growth_std'] if 'terminal_growth_std' in CONFIG else 0.0025
    }

    ticker = CONFIG['ticker']

    # Simple Model
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

    # Enhanced Model: Bull Case
    bull_model = DCFModel(
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
        reinvestment_cap=inputs['reinvestment_cap'],
        scenario='bull'
    )

    bull_results = run_monte_carlo(
        bull_model,
        num_simulations=CONFIG['num_simulations'],
        wacc_std=inputs['wacc_std'],
        high_growth_std=inputs['high_growth_std'],
        terminal_growth_std=inputs['terminal_growth_std'],
        fcf_error=inputs['fcf_error'],
        net_debt_error=CONFIG['net_debt_error']
    )

    # Enhanced Model: Doomsday Scenario
    doomsday_model = DCFModel(
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
        reinvestment_cap=inputs['reinvestment_cap'],
        scenario='doomsday'
    )

    doomsday_results = run_monte_carlo(
        doomsday_model,
        num_simulations=CONFIG['num_simulations'],
        wacc_std=inputs['wacc_std'],
        high_growth_std=inputs['high_growth_std'],
        terminal_growth_std=inputs['terminal_growth_std'],
        fcf_error=inputs['fcf_error'],
        net_debt_error=CONFIG['net_debt_error']
    )

    # Plot all results
    plot_simulation_results(
        simple_results=simple_results,
        optimal_results=bull_results,
        doomsday_results=doomsday_results,
        current_price=inputs['current_price'],
        ticker=ticker,
        currency=inputs['currency'],
        inputs=inputs,
        logger=logger
    )

    # Calculate and log results
    mean_simple = np.mean(simple_results)
    std_simple = np.std(simple_results)
    mean_bull = np.mean(bull_results)
    std_bull = np.std(bull_results)
    mean_doomsday = np.mean(doomsday_results)
    std_doomsday = np.std(doomsday_results)

    logger.info("DCF Monte Carlo simulation completed")

    logger.info("\nSimple Model Results for JPM:")
    logger.info(f"Mean Fair Value: {mean_simple:.2f} USD")
    logger.info(f"1σ Range: {mean_simple - std_simple:.2f} to {mean_simple + std_simple:.2f}")
    logger.info(f"2σ Range: {mean_simple - 2 * std_simple:.2f} to {mean_simple + 2 * std_simple:.2f}")
    logger.info(f"Current Price: {inputs['current_price']:.2f} USD")

    logger.info("\nEnhanced Model (Bull) Results for JPM:")
    logger.info(f"Mean Fair Value: {mean_bull:.2f} USD")
    logger.info(f"1σ Range: {mean_bull - std_bull:.2f} to {mean_bull + std_bull:.2f}")
    logger.info(f"2σ Range: {mean_bull - 2 * std_bull:.2f} to {mean_bull + 2 * std_bull:.2f}")

    logger.info("\nEnhanced Model (Doomsday) Results for JPM:")
    logger.info(f"Mean Fair Value: {mean_doomsday:.2f} USD")
    logger.info(f"1σ Range: {mean_doomsday - std_doomsday:.2f} to {mean_doomsday + std_doomsday:.2f}")
    logger.info(f"2σ Range: {mean_doomsday - 2 * std_doomsday:.2f} to {mean_doomsday + 2 * std_doomsday:.2f}")

    # Perform sensitivity analysis
    from sensitivity_analysis import perform_sensitivity_analysis
    perform_sensitivity_analysis(
        inputs=inputs,
        mean_val=mean_simple,
        model_type='Simple',
        ticker=CONFIG['ticker']
    )
    perform_sensitivity_analysis(
        inputs=inputs,
        mean_val=mean_bull,
        model_type='Enhanced',
        ticker=CONFIG['ticker']
    )
