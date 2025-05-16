import numpy as np
import multiprocessing
from functools import lru_cache
from dcf_model import DCFModel
from config import CONFIG

def run_monte_carlo(base_model, num_simulations, wacc_std, high_growth_std, terminal_growth_std, fcf_error, net_debt_error):
    results = []
    failed_simulations = 0
    for i in range(num_simulations):
        try:
            wacc = np.random.normal(base_model.wacc, wacc_std)
            #wacc = max(wacc, CONFIG['wacc_min'])
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
                reinvestment_cap=base_model.reinvestment_cap,
                scenario=base_model.scenario
            )
            fair_value = model.calculate_dcf()
            if fair_value >= 0:
                results.append(fair_value)
        except Exception as e:
            failed_simulations += 1
            logger.debug(f"Simulation {i} failed: {str(e)}")
    if failed_simulations > 0:
        logger.warning(f"{failed_simulations} simulations failed")
    # Ensure dtype=object to handle ragged nested sequences
    results = np.array(results, dtype=object)
    return np.array(results)

# Vectorize and parallelize Monte Carlo simulations
@lru_cache(maxsize=None)
def run_monte_carlo_parallel(base_model, num_simulations, wacc_std, high_growth_std, terminal_growth_std, fcf_error, net_debt_error):
    def simulate(_):
        try:
            wacc = np.random.normal(base_model.wacc, wacc_std)
            #wacc = max(wacc, CONFIG['wacc_min'])
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
                reinvestment_cap=base_model.reinvestment_cap,
                scenario=base_model.scenario
            )
            return model.calculate_dcf()
        except Exception:
            return None

    with multiprocessing.Pool() as pool:
        results = pool.map(simulate, range(num_simulations))

    # Filter out None results and return as a NumPy array
    return np.array([r for r in results if r is not None], dtype=object)