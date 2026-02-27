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
    ord_close_to_usd_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'ordinary', 'ord_close_to_usd_adr_PX_LAST_adjust_none.csv')
    betas_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'models', 'ordinary_betas_index_only.csv')
    futures_dir = os.path.join(SCRIPT_DIR, '../data/processed/futures/converted_bbo')
    adr_nbbo_dir = os.path.join(SCRIPT_DIR, '../data/raw/adrs/bbo-1m/nbbo')
    adr_info_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    futures_symbols_filename = os.path.join(SCRIPT_DIR, '../data/raw/futures_symbols.csv')
    output_dir = os.path.join(SCRIPT_DIR, f'..', 'data', 'processed', 'db_futures_only_signal')

    # Read ADR info
    adr_info = pd.read_csv(adr_info_filename)
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity','')
    adr_tickers = adr_info['adr'].tolist()
    adr_dict = dict(
        zip(adr_info["id"], adr_info["adr"].str.replace(' US Equity', ''))
    )

    # reading daily data
    adr_domestic_close = pd.read_csv(domestic_close_mid_filename, index_col=0)
    ord_close_to_usd = pd.read_csv(ord_close_to_usd_filename, index_col=0).rename(columns=adr_dict)
    # using different adr baseline for asia
    asia_exchanges = {'XTKS', 'XASX', 'XHKG', 'XSES', 'XSHG', 'XSHE'}
    asia_tickers = adr_info[adr_info['exchange'].isin(asia_exchanges)]['adr'].str.replace(' US Equity','').tolist()
    available_asia_tickers = [t for t in asia_tickers if t in ord_close_to_usd.columns]
    adr_theo_start_asia = ord_close_to_usd[available_asia_tickers]
    adr_domestic_close = pd.concat([adr_domestic_close.drop(columns=available_asia_tickers, errors='ignore'), adr_theo_start_asia], axis=1).sort_index()
    params = utils.load_params()
    
    start_date = params['start_date']
    end_date = params['end_date']
    print("Reading Futures data...")
    df = pd.read_parquet(futures_dir,
                        filters=[('timestamp','>=', pd.Timestamp(start_date, tz='America/New_York'))],
                        columns=['timestamp','symbol','close'])
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df = df.set_index('timestamp')

    close_offsets = pd.read_csv(os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'close_time_offsets.csv'))
    time_futures_after_close = {
        row['exchange_mic']: pd.Timedelta(str(row['offset']))
        for _, row in close_offsets.iterrows()
    }
    betas = pd.read_csv(betas_filename, index_col=0)
    
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
    missing_offsets = sorted(set(exchanges) - set(time_futures_after_close.keys()))
    if missing_offsets:
        raise RuntimeError(
            "Missing close-time offsets for exchanges in data/raw/close_time_offsets.csv: "
            + ", ".join(missing_offsets)
        )
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
    merged_adr_info = adr_info.merge(futures_symbols, left_on='index_future_bbg', right_on='bloomberg_symbol')
    stock_to_index = merged_adr_info.set_index(merged_adr_info['adr'].str.replace(' US Equity', ''))['first_rate_symbol'].to_dict()

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
        adr_df = pd.read_parquet(adr_nbbo_dir,
                                    filters=[('ticker','==', ticker)],
                                    columns=['ticker','date','nbbo_bid','nbbo_ask'])
        adr_df['mid'] = (adr_df['nbbo_bid'] + adr_df['nbbo_ask']) / 2
        
        if adr_df.empty:
            print(f"No ADR NBBO data for {ticker}, skipping...")
            continue

        merged_adr = adr_df.merge(adr_domestic_close[ticker].rename('adr_domestic_close'), left_on='date', right_index=True)
        merged_adr['adr_ret'] = ((merged_adr['mid'] - merged_adr['adr_domestic_close']) / merged_adr['adr_domestic_close']).to_frame(name='adr_ret')
        merged_adr = merged_adr.merge(close_df, left_on='date', right_index=True)
        adr_ret = (merged_adr[merged_adr.index >= merged_adr['domestic_close_time'] + 
                              time_futures_after_close[exchange]]['adr_ret'])
        
        # Get futures price at domestic close
        fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time','close']].apply(
            lambda x: x[x.index <= x['domestic_close_time'] + time_futures_after_close[exchange]].iloc[-1]['close'] if (x.index <= x['domestic_close_time'] + time_futures_after_close[exchange]).any() else np.nan
        ).to_frame(name='fut_domestic_close')
        
        # Filter futures data to only timestamps after domestic close
        merged_fut_after_close = merged_fut[merged_fut.index > merged_fut['domestic_close_time']].copy()

        # Merge with domestic close futures price
        merged_fut_after_close = merged_fut_after_close.merge(
            fut_domestic_close, left_on='date', right_index=True
        )
        
        # Calculate returns at each timestamp
        merged_fut_after_close['fut_ret'] = (
            (merged_fut_after_close['close'] - merged_fut_after_close['fut_domestic_close']) / 
            merged_fut_after_close['fut_domestic_close']
        )

        # Merge with betas
        beta_value = betas.loc[:, ticker] if ticker in betas.columns else None
        if beta_value is None:
            print(f"No beta for {ticker}, skipping...")
            continue
        
        merged_fut_after_close = merged_fut_after_close.merge(
            beta_value.rename('beta'), left_on='date', right_index=True
        )

        merged_all = merged_fut_after_close.merge(merged_adr[['adr_ret']], 
                                                  left_index=True, right_index=True)
        
        # Calculate signal: futures return * beta - ADR return
        merged_all['signal'] = merged_all['fut_ret'] * merged_all['beta'] - merged_all['adr_ret']
        merged_all['date'] = merged_all.index.strftime('%Y-%m-%d')

        # Keep only relevant columns
        signal_df = (merged_all[['signal','date']].copy().groupby('date')[['signal']]
                     .apply(lambda _df: _df[['signal']].resample('1min').first().ffill())
                     .droplevel(0))
        signal_df['date'] = signal_df.index.strftime('%Y-%m-%d')
        # Save this ticker's data as parquet
        ticker_output_path = os.path.join(output_dir, f'ticker={ticker}')
        os.makedirs(ticker_output_path, exist_ok=True)
        output_filename = os.path.join(ticker_output_path, f'data.parquet')
        signal_df.to_parquet(output_filename)
        
        print(f"Processed and saved signal for {ticker}")

    print(f"Saved partitioned parquet dataset to {output_dir}")
