import os
import pandas as pd
from config import tickers, download_config  # Import tickers and download_config

def convert_to_usd(input_folder, output_folder, conversion_rates):
    """
    Convert index data to USD using provided conversion rates.

    Parameters:
        input_folder (str): Path to the folder containing index data files.
        output_folder (str): Path to the folder where USD-converted files will be saved.
        conversion_rates (dict): Dictionary with country codes as keys and conversion rates as values.
    """
    os.makedirs(output_folder, exist_ok=True)

    for country, ticker in tickers.items():
        file_name = f"{country}_index_data.csv"
        file_path = os.path.join(input_folder, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Ensure all data is numeric
            df = df.apply(pd.to_numeric, errors='coerce')

            # Convert to USD
            conversion_rate = conversion_rates.get(country, 1.0)  # Default to 1.0 if no rate is provided
            df_usd = df * conversion_rate

            # Save the converted data
            output_file = os.path.join(output_folder, f"{country}_usd_index_data.csv")
            df_usd.to_csv(output_file)
            print(f"Converted {file_name} to USD and saved as {output_file}")
        else:
            print(f"Warning: {file_name} not found in {input_folder}. Skipping.")

if __name__ == "__main__":
    input_folder = "index_data"
    output_folder = "usd_index_data"

    # Example conversion rates (replace with actual rates or fetch dynamically if needed)
    conversion_rates = {
        "US": 1.0,       # USD to USD
        "Germany": 1.1,  # EUR to USD
        "China": 0.15,   # CNY to USD
        "India": 0.012,  # INR to USD
        "Japan": 0.007   # JPY to USD
    }

    convert_to_usd(input_folder, output_folder, conversion_rates)