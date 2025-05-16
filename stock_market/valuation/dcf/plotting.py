import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import iqr

def plot_simulation_results(simple_results, optimal_results, doomsday_results, current_price, ticker, currency, inputs, logger):
    if len(simple_results) == 0 and len(optimal_results) == 0 and len(doomsday_results) == 0:
        print(f"Error: No valid simulation results for {ticker}. Check input data.")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))

    # Filter outliers using percentile-based capping (e.g., 99th percentile)
    def filter_outliers(data, percentile=99):
        if len(data) == 0:
            return data
        upper_limit = np.percentile(data, percentile)
        return data[data <= upper_limit]

    # Apply outlier filtering
    simple_filtered = filter_outliers(simple_results)
    optimal_filtered = filter_outliers(optimal_results)
    doomsday_filtered = filter_outliers(doomsday_results)

    # Combine filtered results to calculate bin width
    all_results = np.concatenate([
        simple_filtered if len(simple_filtered) > 0 else [],
        optimal_filtered if len(optimal_filtered) > 0 else [],
        doomsday_filtered if len(doomsday_filtered) > 0 else []
    ])

    if len(all_results) > 0:
        # Use Freedman-Diaconis rule for bin width
        q75, q25 = np.percentile(all_results, [75, 25])
        bin_width = 2 * (q75 - q25) * len(all_results) ** (-1/3)  # Freedman-Diaconis
        data_range = max(all_results) - min(all_results)
        bins = int(data_range / bin_width) if bin_width > 0 else 200  # Fallback to 200 bins
        bins = min(bins, 200)  # Cap bins

        # Plot Simple Model
        if len(simple_filtered) > 0:
            sns.histplot(simple_filtered, bins=bins, kde=True, color='skyblue', label='Simple Model', alpha=0.3)
            mean_simple = np.mean(simple_filtered)
            plt.axvline(mean_simple, color='blue', linestyle='--', label=f'Simple Mean: {mean_simple:.2f}')

        # Plot Enhanced Model: Optimal
        if len(optimal_filtered) > 0:
            sns.histplot(optimal_filtered, bins=bins, kde=True, color='green', label='Enhanced (Optimal)', alpha=0.3)
            mean_optimal = np.mean(optimal_filtered)
            plt.axvline(mean_optimal, color='darkgreen', linestyle='--', label=f'Optimal Mean: {mean_optimal:.2f}')

        # Plot Enhanced Model: Doomsday
        if len(doomsday_filtered) > 0:
            sns.histplot(doomsday_filtered, bins=bins, kde=True, color='red', label='Enhanced (Doomsday)', alpha=0.3)
            mean_doomsday = np.mean(doomsday_filtered)
            plt.axvline(mean_doomsday, color='darkred', linestyle='--', label=f'Doomsday Mean: {mean_doomsday:.2f}')

        # Plot Current Price
        if current_price:
            plt.axvline(current_price, color='black', linestyle='-', label=f'Current Price: {current_price:.2f}')

        # Limit x-axis to a reasonable range (e.g., 0 to 500 USD) for better visualization
        plt.xlim(0, 500)

    plt.title(f'DCF Monte Carlo Simulation Results for {ticker}')
    plt.xlabel(f'Fair Value per Share ({currency})')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('simulation_results_cleaned.png')

    # Log parameter comparison
    log_parameter_comparison(inputs, logger)

def log_parameter_comparison(inputs, logger):
    """Logs the parameter values for Base, Best, and Worst cases."""
    logger.info("\nParameter Comparison for Scenarios:")
    logger.info("{:<20} {:<15} {:<15} {:<15}".format("Parameter", "Base Case", "Best Case", "Worst Case"))
    logger.info("-" * 65)

    parameters = [
        ("WACC", inputs['wacc'], inputs['wacc'] * 0.625, inputs['wacc'] * 1.25),
        ("High Growth Rate", inputs['high_growth_rate'], inputs['high_growth_rate'] * 3.0, inputs['high_growth_rate'] * 0.125),
        ("Terminal Growth Rate", inputs['terminal_growth_rate'], min(0.04, inputs['wacc'] - 0.01), min(0.015, inputs['wacc'] - 0.01)),
        ("Margin Compression", 0, 0, inputs['margin_compression'] * 5.0),
        ("Reinvestment Cap", inputs['reinvestment_cap'], 0, 0.05),
    ]

    for param, base, best, worst in parameters:
        logger.info("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(param, base, best, worst))

# Collect parameter sets within 20% of the real price
def collect_accepted_parameters(simulation_results, real_price, acceptance_threshold, logger):
    # Ensure simulation_results is a list of dictionaries with 'prices' and 'parameters' keys
    if isinstance(simulation_results, np.ndarray):
        simulation_results = [{'prices': [price], 'parameters': None} for price in simulation_results]

    # Add a check to ensure the structure of simulation_results is as expected
    if not isinstance(simulation_results, list) or not all(isinstance(res, dict) and 'prices' in res for res in simulation_results):
        logger.error("Invalid structure for simulation_results. Expected a list of dictionaries with 'prices' and 'parameters' keys.")
        return []

    accepted_parameters = []
    for result in simulation_results:
        mean_price = np.mean(result['prices'])
        if abs(mean_price - real_price) / real_price <= acceptance_threshold:
            accepted_parameters.append(result['parameters'])
    return accepted_parameters

# Plot the Monte Carlo distribution of accepted parameter sets
def plot_mc_distribution(accepted_parameters, simulation_results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for params in accepted_parameters:
        prices = [res['prices'] for res in simulation_results if res['parameters'] == params]
        for price_set in prices:
            plt.hist(price_set, bins=50, alpha=0.5, label=f"Params: {params}")

    plt.title("Monte Carlo Distribution of Accepted Parameter Sets")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()