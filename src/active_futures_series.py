import os
import pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    fut_bbo_dir = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'bbo-1m')
    futures_symbol_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures_symbols.csv')
    adr_info_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    adr_info = pd.read_csv(adr_info_filename)
    futures_symbols = pd.read_csv(futures_symbol_filename)

    futures_map = dict(zip(futures_symbols['bloomberg_symbol'], futures_symbols['exchange_symbol']))

    bbg_futures = adr_info['index_future_bbg'].unique().tolist()
    exchange_futures = [futures_map[bbg] for bbg in bbg_futures if bbg in futures_map]

    