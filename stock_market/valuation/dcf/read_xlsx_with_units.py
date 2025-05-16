import os
import pandas as pd
import logging
import glob



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

def read_xlsx_files_with_units(ticker, data_dir, show_output=True):
    data_dir = os.path.join(data_dir, ticker)
    xlsx_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    results = {}
    for file_path in xlsx_files:
        file_lower = os.path.basename(file_path).lower()
        if "income" in file_lower:
            results['income'] = pd.read_excel(file_path, sheet_name=0, header=0, index_col=0)
            if show_output:
                print(f"Loaded Income file: {file_path}")
        elif "balance" in file_lower:
            results['balance'] = pd.read_excel(file_path, sheet_name=0, header=0, index_col=0)
            if show_output:
                print(f"Loaded Balance file: {file_path}")
        elif "cash flow" in file_lower:
            results['ratios'] = pd.read_excel(file_path, sheet_name=0, header=0, index_col=0)
            if show_output:
                print(f"Loaded Cash Flow file: {file_path}")
        elif "ratio" in file_lower:
            results['cashflow'] = pd.read_excel(file_path, sheet_name=0, header=0, index_col=0)
            if show_output:
                print(f"Loaded Ratios file: {file_path}")
        else:
            # Optionally load any other .xlsx files
            results[file_lower] = pd.read_excel(file_path, sheet_name=0, header=0, index_col=0)
            if show_output:
                print(f"Loaded Other file: {file_path}")
    if show_output:
        for key, df in results.items():
            print(f"{key} DataFrame shape: {df.shape}")
    return results



if __name__ == "__main__":
    ticker = "NKE"
    data_dir = "../data"
    read_xlsx_files_with_units(ticker, data_dir, show_output=True)