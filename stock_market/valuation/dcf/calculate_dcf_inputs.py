import logging

def calculate_dcf_inputs(data_frames, ticker):
    def detect_units(df, context=""):
        scaling_factors = []
        for label in list(df.columns) + list(df.index):
            label_str = str(label).lower()
            if 'million' in label_str or 'mm' in label_str:
                scaling_factors.append(1e6)
            elif 'thousand' in label_str or 'k' in label_str:
                scaling_factors.append(1e3)
            elif 'billion' in label_str or 'bn' in label_str:
                scaling_factors.append(1e9)
        scaling_factor = max(scaling_factors, default=1e6)
        logging.info(f"Detected units for {ticker} {context}: scaling factor {scaling_factor}")
        return scaling_factor

    scaling_factors = {key: detect_units(df, key) for key, df in data_frames.items()}
    max_scaling_factor = max(scaling_factors.values())
    logging.info(f"Using consistent scaling factor for {ticker}: {max_scaling_factor}")

    # Placeholder: Replace with actual extraction logic
    inputs = {
        'starting_fcf': 1000e6,
        'net_debt': 500e6,
        'shares_outstanding': 100e6,
        'current_price': 150,
        'roic': 0.15,
    }
    return inputs
