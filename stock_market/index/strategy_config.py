# config.py

# Weights configuration for different time periods
WEIGHTS_BY_YEAR = {
    'before 1990': {'US': 0.3, 'Germany': 0.3, 'Japan': 0.4, 'China': 0.0, 'India': 0.0},
    'before 1998': {'US': 0.2, 'Germany': 0.3, 'Japan': 0.5, 'China': 0.0, 'India': 0.0},
    'before 2000': {'US': 0.3, 'Germany': 0.3, 'Japan': 0.3, 'China': 0.1, 'India': 0.0},
    'before 2005': {'US': 0.3, 'Germany': 0.2, 'Japan': 0.2, 'China': 0.15, 'India': 0.15},
    float('inf'): {'US': 0.4, 'Germany': 0.2, 'Japan': 0.05, 'China': 0.15, 'India': 0.2},
}

# SIP configuration
sip_config = {
    "start_date": "1980-01-01",  # Start date for SIP investment
    "end_date": "2022-12-31",    # End date for SIP investment
    "monthly_investment": 1.0    # Monthly investment amount in USD
}

# Data file paths
INDEX_PATHS = {
    'US': 'usd_index_data/US_usd_index_data.csv',
    'Germany': 'usd_index_data/Germany_usd_index_data.csv',
    'Japan': 'usd_index_data/Japan_usd_index_data.csv',
    'China': 'usd_index_data/China_usd_index_data.csv',
    'India': 'usd_index_data/India_usd_index_data.csv'
}
