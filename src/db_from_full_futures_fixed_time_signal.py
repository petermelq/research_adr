import os
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import pandas_market_calendars as mcal
import utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    signal_dir = os.path.join(SCRIPT_DIR, '../data/processed/db_futures_only_signal')
    adr_info_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    output_filename = os.path.join(SCRIPT_DIR, f'..', 'data', 'processed', 'fixed_time_signal.csv')

    # Read ADR info
    adr_info = pd.read_csv(adr_info_filename)
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity','')
    adr_tickers = adr_info['adr'].tolist()
    
    params = utils.load_params()
    time_to_save = dt.time(params['fixed_trade_time_hours'], 
                            params['fixed_trade_time_min'])

    signal_df = pd.read_parquet(signal_dir)
    all_signal = {}
    for ticker in adr_tickers:
        ticker_signal_df = signal_df[signal_df['ticker'] == ticker]
        all_signal[ticker] = ticker_signal_df.between_time('0:00',time_to_save).groupby('date', observed=True)['signal'].last()
        print(f"Processed signal for {ticker}")

    all_signal_df = pd.DataFrame(all_signal)
    all_signal_df.to_csv(output_filename)