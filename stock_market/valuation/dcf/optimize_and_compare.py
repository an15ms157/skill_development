import os
import numpy as np
from scipy.optimize import differential_evolution, minimize
from config import CONFIG
from dcf_model import DCFModel
from monte_carlo import run_monte_carlo
import yfinance as yf
import json
from plotting import collect_accepted_parameters, plot_mc_distribution
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache

# Create output directory if it doesn't exist
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def get_current_price(ticker):
    """Fetch the current stock price from Yahoo Finance with robust retry logic."""
    stock = yf.Ticker(ticker)
    history = stock.history(period="7d")
    if 'Close' in history.columns and not history['Close'].empty:
        return history['Close'].iloc[-1]
    raise ValueError("No valid data in 'Close' column.")

def optimize_parameters(target_price, inputs, bounds, logger, run_number):
    """Optimize DCF parameters to match MC mean price to target price, stopping if within tolerance level."""
    logger.debug(f"Run {run_number}: Parameter bounds: {bounds}")

    # Cache Monte Carlo results to avoid redundant simulations
    @lru_cache(maxsize=100)
    def cached_monte_carlo(wacc, high_growth_rate, terminal_growth_rate, growth_decay, margin_compression, reinvestment_cap, high_growth_years):
        return run_monte_carlo(
            base_model=DCFModel(
                starting_fcf=inputs['starting_fcf'],
                wacc=wacc,
                high_growth_rate=high_growth_rate,
                terminal_growth_rate=terminal_growth_rate,
                net_debt=inputs['net_debt'],
                shares_outstanding=inputs['shares_outstanding'],
                high_growth_years=int(high_growth_years),  # Ensure integer value
                growth_decay=growth_decay,
                margin_compression=margin_compression,
                roic=inputs['roic'],
                wacc_drift_std=inputs['wacc_drift_std'],
                use_enhanced_model=False,
                ticker=CONFIG['ticker'],
                reinvestment_cap=reinvestment_cap
            ),
            num_simulations=CONFIG['num_simulations'],
            wacc_std=wacc * CONFIG['volatility_factor'],
            high_growth_std=high_growth_rate * CONFIG['volatility_factor'] * 2,
            terminal_growth_std=terminal_growth_rate * CONFIG['volatility_factor'],
            fcf_error=CONFIG['fcf_error'],
            net_debt_error=CONFIG['net_debt_error']
        )

    def objective(params, normalize=False):
        if normalize:
            params = [low + param * (high - low) for param, (low, high) in zip(params, bounds)]
        wacc, high_growth_rate, terminal_growth_rate, growth_decay, margin_compression, reinvestment_cap, high_growth_years = params

        # Economic plausibility checks
        #if terminal_growth_rate >= wacc:
        #    logger.debug(f"Run {run_number}: Penalizing: terminal_growth_rate ({terminal_growth_rate}) >= wacc ({wacc})")
        #    return float('inf')
        #if growth_decay >= high_growth_rate:
        #    logger.debug(f"Run {run_number}: Penalizing: growth_decay ({growth_decay}) >= high_growth_rate ({high_growth_rate})")
        #    return float('inf')
        #if terminal_growth_rate >= high_growth_rate:
        #    logger.debug(f"Run {run_number}: Penalizing: terminal_growth_rate ({terminal_growth_rate}) >= high_growth_rate ({high_growth_rate})")
        #    return float('inf')

        inputs['wacc'], inputs['high_growth_rate'], inputs['terminal_growth_rate'], \
        inputs['growth_decay'], inputs['margin_compression'], inputs['reinvestment_cap'], inputs['high_growth_years'] = params
        simulation_results = cached_monte_carlo(
            wacc, high_growth_rate, terminal_growth_rate, growth_decay, margin_compression, reinvestment_cap, high_growth_years
        )
        if len(simulation_results) > 0:
            simulation_avg = np.mean(simulation_results)
            logger.debug(f"Run {run_number}: Params: {params}, Simulation average: {simulation_avg}")
            return (simulation_avg - target_price) ** 2
        logger.error(f"Run {run_number}: Monte Carlo simulation returned no results.")
        return float('inf')

    # Track best parameters for early stopping
    best_params = None
    best_rmse = float('inf')

    # Callback to stop optimization if within tolerance level of target price
    def stop_early(xk, convergence):
        nonlocal best_params, best_rmse
        obj_value = np.sqrt(objective(xk))  # RMSE
        TOLERANCE_PERCENT = CONFIG['tolerance_level'] * 100
        if obj_value / target_price <= CONFIG['tolerance_level']:
            best_params = xk
            best_rmse = obj_value
            logger.info(f"Run {run_number}: Stopping optimization: MC mean within {TOLERANCE_PERCENT}% of target price (RMSE: {obj_value}, Params: {xk})")
            return True
        if obj_value < best_rmse:
            best_params = xk
            best_rmse = obj_value
        logger.debug(f"Run {run_number}: Iteration: {convergence}, Best params: {xk}, RMSE: {obj_value}")
        return False

    # Try differential evolution
    logger.info(f"Run {run_number}: Starting differential evolution optimization...")
    result = differential_evolution(
        objective, bounds, maxiter=1000, popsize=30, disp=True, callback=stop_early
    )
    if result.success or best_params is not None:
        optimized_params = best_params if best_params is not None else result.x
        logger.info(f"Run {run_number}: Optimization succeeded. Optimized parameters: {optimized_params}")
        # Log market-implied expectations
        logger.info(f"Run {run_number}: Market-implied expectations: WACC={optimized_params[0]:.2%}, "
                    f"High Growth Rate={optimized_params[1]:.2%}, "
                    f"Terminal Growth Rate={optimized_params[2]:.2%}, "
                    f"Growth Decay={optimized_params[3]:.2%}, "
                    f"Margin Compression={optimized_params[4]:.2%}, "
                    f"Reinvestment Cap={optimized_params[5]:.2%}, "
                    f"High Growth Years={int(optimized_params[6])}")
        return optimized_params

    # Fallback to L-BFGS-B
    logger.warning(f"Run {run_number}: Differential evolution failed: {result.message}. Falling back to L-BFGS-B.")
    result = minimize(objective, x0=[0.5] * len(bounds), bounds=bounds, method='L-BFGS-B')
    if result.success:
        optimized_params = [low + param * (high - low) for param, (low, high) in zip(result.x, bounds)]
        logger.info(f"Run {run_number}: L-BFGS-B succeeded. Optimized parameters: {optimized_params}")
        logger.info(f"Run {run_number}: Market-implied expectations: WACC={optimized_params[0]:.2%}, "
                    f"High Growth Rate={optimized_params[1]:.2%}, "
                    f"Terminal Growth Rate={optimized_params[2]:.2%}, "
                    f"Growth Decay={optimized_params[3]:.2%}, "
                    f"Margin Compression={optimized_params[4]:.2%}, "
                    f"Reinvestment Cap={optimized_params[5]:.2%}, "
                    f"High Growth Years={int(optimized_params[6])}")
        return optimized_params
    logger.error(f"Run {run_number}: L-BFGS-B failed: {result.message}")
    return None

def save_accepted_parameters(all_accepted_parameters):
    """Save all accepted parameters to a single file in the output directory."""
    output_file = os.path.join(output_dir, 'accepted_parameters.json')
    with open(output_file, 'w') as f:
        json.dump(all_accepted_parameters, f, indent=4)
    logger.info(f"Saved {len(all_accepted_parameters)} accepted parameter sets to {output_file}")

if __name__ == "__main__":
    from config import logger

    # Fetch the target price
    ticker = CONFIG['ticker']
    try:
        target_price = get_current_price(ticker)
        logger.info(f"Fetched Target Price for {ticker}: {target_price:.2f} USD")
    except Exception as e:
        logger.error(f"Failed to fetch price for {ticker}: {e}. Using default price from CONFIG.")
        target_price = CONFIG['default_price_per_share']
        logger.info(f"Using default price for {ticker}: {target_price:.2f} USD. Consider using an alternative API (e.g., Alpha Vantage) for reliable price data.")

    # Inputs derived from config
    inputs = {
        'starting_fcf': CONFIG['default_fcf'],
        'wacc': CONFIG['default_wacc'],
        'high_growth_rate': 0.04,
        'terminal_growth_rate': CONFIG['gdp_growth'],
        'net_debt': CONFIG['default_net_debt'],
        'shares_outstanding': CONFIG['default_shares_outstanding'],
        'growth_decay': CONFIG['growth_decay'],
        'margin_compression': CONFIG['default_margin_compression'],
        'roic': CONFIG['default_roic'],
        'wacc_drift_std': CONFIG['default_wacc_drift_std'],
        'reinvestment_cap': CONFIG['reinvestment_cap'],
        'high_growth_years': CONFIG['high_growth_years']
    }

    # Run optimization N=10 times
    N = CONFIG['RUNS']
    tolerance = CONFIG['tolerance_level']
    all_accepted_parameters = []
    for run_number in range(1, N + 1):
        logger.info(f"Starting optimization run {run_number} of {N}")
        optimized_params = optimize_parameters(target_price, inputs, CONFIG['parameter_bounds'], logger, run_number)

        if optimized_params is not None:
            logger.info(f"Run {run_number}: Optimized Parameters: {optimized_params}")

            # Run final Monte Carlo simulation with more simulations for accuracy
            simulation_results = run_monte_carlo(
                base_model=DCFModel(
                    starting_fcf=inputs['starting_fcf'],
                    wacc=optimized_params[0],
                    high_growth_rate=optimized_params[1],
                    terminal_growth_rate=optimized_params[2],
                    net_debt=inputs['net_debt'],
                    shares_outstanding=inputs['shares_outstanding'],
                    high_growth_years=int(optimized_params[6]),  # Ensure integer value
                    growth_decay=optimized_params[3],
                    margin_compression=optimized_params[4],
                    roic=inputs['roic'],
                    wacc_drift_std=inputs['wacc_drift_std'],
                    use_enhanced_model=False,
                    ticker=CONFIG['ticker'],
                    reinvestment_cap=optimized_params[5]
                ),
                num_simulations=200,  # Increased for final simulation
                wacc_std=optimized_params[0] * CONFIG['volatility_factor'],
                high_growth_std=optimized_params[1] * CONFIG['volatility_factor'] * 2,
                terminal_growth_std=optimized_params[2] * CONFIG['volatility_factor'],
                fcf_error=CONFIG['fcf_error'],
                net_debt_error=CONFIG['net_debt_error']
            )

            # Check if final MC mean is within tolerance level of target price
            simulation_mean = np.mean(simulation_results)
            TOLERANCE_PERCENT = CONFIG['tolerance_level'] * 100
            logger.info(f"Run {run_number}: Final MC simulation mean: {simulation_mean:.2f} USD, Target price: {target_price:.2f} USD")
            if abs(simulation_mean - target_price) / target_price <= tolerance:
                accepted_params = {
                    'run_number': run_number,
                    'wacc': optimized_params[0],
                    'high_growth_rate': optimized_params[1],
                    'terminal_growth_rate': optimized_params[2],
                    'growth_decay': optimized_params[3],
                    'margin_compression': optimized_params[4],
                    'reinvestment_cap': optimized_params[5],
                    'high_growth_years': int(optimized_params[6]),  # Ensure integer value
                    'mc_mean': simulation_mean,
                    'target_price': target_price
                }

                # Save accepted parameters dynamically after each run
                output_file = os.path.join(output_dir, 'accepted_parameters.json')
                if os.path.exists(output_file):
                    # Add error handling for empty or invalid JSON files
                    try:
                        with open(output_file, 'r') as f:
                            if os.stat(output_file).st_size == 0:  # Check if the file is empty
                                all_accepted_parameters = []
                            else:
                                all_accepted_parameters = json.load(f)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON format in {output_file}. Initializing as an empty list.")
                        all_accepted_parameters = []
                else:
                    all_accepted_parameters = []

                all_accepted_parameters.append(accepted_params)

                with open(output_file, 'w') as f:
                    json.dump(all_accepted_parameters, f, indent=4)
                logger.info(f"Run {run_number}: Parameters dynamically saved to {output_file}")

    # Save all accepted parameters
    output_file = os.path.join(output_dir, 'accepted_parameters.json')
    with open(output_file, 'w') as f:
        json.dump(all_accepted_parameters, f, indent=4)
    logger.info(f"All accepted parameters saved to {output_file}")