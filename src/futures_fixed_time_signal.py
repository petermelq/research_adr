import os
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import pandas_market_calendars as mcal
import utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    domestic_close_mid_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'adrs', 'adr_mid_at_ord_auction_adjust_none.csv')
    afternoon_mid_close_time = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'adrs', 'adr_daily_fixed_time_mid.csv')
    betas_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'models', 'ordinary_betas_index_only.csv')
    futures_dir = os.path.join(SCRIPT_DIR, '../data/processed/futures/converted_minute_bars')
    adr_info_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    futures_symbols_filename = os.path.join(SCRIPT_DIR, '../data/raw/futures_symbols.csv')
    output_filename = os.path.join(SCRIPT_DIR, f'..', 'data', 'processed', 'fixed_time_signal.csv')

    domestic_close_mid = pd.read_csv(domestic_close_mid_filename)
    afternoon_mid_df = pd.read_csv(afternoon_mid_close_time)

    # Fixed time of day to save mid price
    params = utils.load_params()
    time_to_save = dt.time(params['fixed_trade_time_hours'], 
                           params['fixed_trade_time_min'])

    start_date = params['start_date']
    end_date = params['end_date']
    print("Reading Futures data...")
    df = pd.read_parquet(futures_dir,
                        filters=[('timestamp','>=', pd.Timestamp(start_date, tz='America/New_York'))],
                        columns=['timestamp','symbol','close'])
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df = df.set_index('timestamp')
    
    time_futures_after_close = pd.Timedelta(minutes=5)
    betas = pd.read_csv(betas_filename, index_col=0)

    # Fixed time of day to compute signal
    time_to_save = dt.time(13,0)
    start_time = (dt.datetime.combine(dt.date.today(), time_to_save) - pd.Timedelta('30min')).time()

    # Read ADR info
    adr_info = pd.read_csv(adr_info_filename)
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity','')
    adr_tickers = adr_info['adr'].tolist()
    
    # Create close times dataframe
    ny_close_times = (
            mcal.get_calendar('NYSE').schedule(start_date=start_date,
                                                end_date=end_date)['market_close']
                                                .dt.tz_convert('America/New_York')
        )
    ny_close_times.index = ny_close_times.index.astype(str)

    futures_df = df.merge(ny_close_times.rename('ny_market_close_time'), left_on='date', right_index=True)
    futures_df = futures_df[futures_df['ny_market_close_time'].dt.time == dt.time(16,0)]
    
    adr_info = pd.read_csv(adr_info_filename)
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity','')
    adr_tickers = adr_info['adr'].tolist()
    exchanges = adr_info['exchange'].unique().tolist()
    exchange_dict = adr_info.set_index('adr')['exchange'].to_dict()
    
    # Create close times dataframe
    close_times = {}
    for ex in exchanges:
        close_times[ex] = (mcal.get_calendar(ex)
                            .schedule(start_date=start_date,
                            end_date=end_date)['market_close']
                            .dt.tz_convert('America/New_York')
                            ).rename('domestic_close_time')
        close_times[ex].index = close_times[ex].index.astype(str)
    
    all_signal = {}

    futures_symbols = pd.read_csv(futures_symbols_filename)
    futures_to_index = futures_symbols.set_index('exchange_symbol')['first_rate_symbol'].to_dict()
    stock_to_index_future = adr_info.set_index(adr_info['adr'].str.replace(' US Equity',''))['index_future_bbg'].to_dict()
    stock_to_index = {stock: futures_to_index.get(index_future)
                        for stock, index_future in stock_to_index_future.items()}

    for ticker in adr_tickers:
        exchange = exchange_dict[ticker]
        close_df = close_times[exchange]
        futures_symbol = stock_to_index.get(ticker)
        if futures_symbol is None:
            print(f"No futures mapping for {ticker}, skipping...")
            continue
        
        futures_df = df[df['symbol'] == futures_symbol].copy()
        merged_fut = futures_df.merge(close_df, left_on='date', right_index=True)
        
        fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time','close']].apply(
            lambda x: x[x.index <= x['domestic_close_time'] + time_futures_after_close].iloc[-1]['close']
        )
        fut_afternoon = merged_fut.groupby('date')[['close']].apply(
            lambda x: x[x.index.time <= time_to_save].iloc[-1]['close']
        )
        fut_ret = ((fut_afternoon - fut_domestic_close) / fut_domestic_close).to_frame(name='fut_ret')

        merged = fut_ret.merge(betas[ticker].rename('beta'),
                               left_on='date', right_index=True)
        merged['signal'] = merged['fut_ret'] * merged['beta']
        all_signal[ticker] = merged['signal']

        print(f"Processed signal for {ticker}")

    all_signal_df = pd.DataFrame(all_signal)
    all_signal_df.to_csv(output_filename)