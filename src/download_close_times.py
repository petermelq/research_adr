import os
import pandas as pd
from linux_xbbg import blp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Load ordinary tickers from adr_info
    adr_info = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/adr_info.csv'))
    adr_info = adr_info.dropna(subset=['adr'])
    adr_info = adr_info[~adr_info['id'].str.contains(' US Equity')]
    ordinary_tickers = adr_info['id'].tolist()

    # Load index tickers from futures_symbols
    futures_symbols = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/futures_symbols.csv'))
    index_tickers = [f"{idx} Index" for idx in futures_symbols['index'].dropna().unique()]

    # Combine all tickers
    all_tickers = ordinary_tickers + index_tickers

    # Pull BLOOMBERG_CLOSE_TIME
    close_times = blp.bds(all_tickers, 'BLOOMBERG_CLOSE_TIME')
    close_times.columns = ['BLOOMBERG_CLOSE_TIME']

    # Save to CSV
    output_filename = os.path.join(SCRIPT_DIR, '../data/raw/bloomberg_close_times.csv')
    close_times.to_csv(output_filename)
    print(f"Close times saved to {output_filename}")
    print(close_times)
