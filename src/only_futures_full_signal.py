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
    adr_nbbo_dir = os.path.join(SCRIPT_DIR, '../data/raw/adrs/bbo-1m/nbbo')
    adr_info_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    futures_symbols_filename = os.path.join(SCRIPT_DIR, '../data/raw/futures_symbols.csv')
    output_dir = os.path.join(SCRIPT_DIR, f'..', 'data', 'processed', 'fixed_time_signal_parquet')

    params = utils.load_params()
    
    start_date = params['start_date']
    end_date = params['end_date']
    print("Reading Futures data...")
    df = pd.read_parquet(futures_dir,
                        filters=[('timestamp','>=', pd.Timestamp(start_date, tz='America/New_York'))],
                        columns=['timestamp','symbol','close'])
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df = df.set_index('timestamp')
    
    print("Reading ADR NBBO data...")
    adr_nbbo_df = pd.read_parquet(adr_nbbo_dir,
                                   filters=[('timestamp','>=', pd.Timestamp(start_date, tz='America/New_York'))],
                                   columns=['timestamp','symbol','nbbo_bid','nbbo_ask'])
    adr_nbbo_df['mid'] = (adr_nbbo_df['nbbo_bid'] + adr_nbbo_df['nbbo_ask']) / 2
    adr_nbbo_df['date'] = adr_nbbo_df['timestamp'].dt.strftime('%Y-%m-%d')
    adr_nbbo_df = adr_nbbo_df.set_index('timestamp')
    
    time_futures_after_close = {}
    time_futures_after_close['XLON'] = pd.Timedelta('6min')
    time_futures_after_close['XAMS'] = pd.Timedelta('6min')
    time_futures_after_close['XPAR'] = pd.Timedelta('6min')
    time_futures_after_close['XETR'] = pd.Timedelta('6min')
    time_futures_after_close['XMIL'] = pd.Timedelta('6min')
    time_futures_after_close['XBRU'] = pd.Timedelta('6min')
    time_futures_after_close['XMAD'] = pd.Timedelta('6min')
    time_futures_after_close['XHEL'] = pd.Timedelta('0min')
    time_futures_after_close['XDUB'] = pd.Timedelta('0min')
    betas = pd.read_csv(betas_filename, index_col=0)

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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for ticker in adr_tickers:
        exchange = exchange_dict[ticker]
        close_df = close_times[exchange]
        futures_symbol = stock_to_index.get(ticker)
        if futures_symbol is None:
            print(f"No futures mapping for {ticker}, skipping...")
            continue
        
        futures_df = df[df['symbol'] == futures_symbol].copy()
        merged_fut = futures_df.merge(close_df, left_on='date', right_index=True)
        
        # Get ADR NBBO data for this ticker
        adr_df = adr_nbbo_df[adr_nbbo_df['symbol'] == ticker].copy()
        if adr_df.empty:
            print(f"No ADR NBBO data for {ticker}, skipping...")
            continue
        
        merged_adr = adr_df.merge(close_df, left_on='date', right_index=True)
        
        # Get futures price at domestic close
        fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time','close']].apply(
            lambda x: x[x.index <= x['domestic_close_time'] + time_futures_after_close[exchange]].iloc[-1]['close']
        ).to_frame(name='fut_domestic_close')
        
        # Get ADR mid price at domestic close
        adr_domestic_close = merged_adr.groupby('date')[['domestic_close_time','mid']].apply(
            lambda x: x[x.index <= x['domestic_close_time'] + time_futures_after_close[exchange]].iloc[-1]['mid']
        ).to_frame(name='adr_domestic_close')
        
        # Filter futures data to only timestamps after domestic close
        merged_fut_after_close = merged_fut[merged_fut.index > merged_fut['domestic_close_time']].copy()
        
        # Filter ADR data to only timestamps after domestic close
        merged_adr_after_close = merged_adr[merged_adr.index > merged_adr['domestic_close_time']].copy()
        
        # Merge with domestic close futures price
        merged_fut_after_close = merged_fut_after_close.merge(
            fut_domestic_close, left_on='date', right_index=True
        )
        
        # Merge with domestic close ADR price
        merged_adr_after_close = merged_adr_after_close.merge(
            adr_domestic_close, left_on='date', right_index=True
        )
        
        # Calculate returns at each timestamp
        merged_fut_after_close['fut_ret'] = (
            (merged_fut_after_close['close'] - merged_fut_after_close['fut_domestic_close']) / 
            merged_fut_after_close['fut_domestic_close']
        )
        
        # Calculate ADR returns at each timestamp
        merged_adr_after_close['adr_ret'] = (
            (merged_adr_after_close['mid'] - merged_adr_after_close['adr_domestic_close']) / 
            merged_adr_after_close['adr_domestic_close']
        )
        
        # Merge futures and ADR data on timestamp
        merged_after_close = merged_fut_after_close.merge(
            merged_adr_after_close[['adr_ret', 'date']], 
            left_index=True, 
            right_index=True, 
            suffixes=('', '_adr')
        )
        
        # Merge with betas
        beta_value = betas.loc[:, ticker] if ticker in betas.columns else None
        if beta_value is None:
            print(f"No beta for {ticker}, skipping...")
            continue
            
        merged_after_close = merged_after_close.merge(
            beta_value.rename('beta'), left_on='date', right_index=True
        )
        
        # Calculate signal: futures return * beta - ADR return
        merged_after_close['signal'] = merged_after_close['fut_ret'] * merged_after_close['beta'] - merged_after_close['adr_ret']
        merged_after_close['ticker'] = ticker
        
        # Keep only relevant columns
        signal_df = merged_after_close[['ticker', 'date', 'signal']].copy()
        signal_df['timestamp'] = merged_after_close.index
        
        # Save this ticker's data as parquet
        ticker_output_path = os.path.join(output_dir, f'ticker={ticker}')
        os.makedirs(ticker_output_path, exist_ok=True)
        signal_df.to_parquet(os.path.join(ticker_output_path, 'data.parquet'), index=False)
        
        print(f"Processed and saved signal for {ticker}")

    print(f"Saved partitioned parquet dataset to {output_dir}")