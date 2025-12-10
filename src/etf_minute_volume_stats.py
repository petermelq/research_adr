import os
import pandas as pd
import pandas_market_calendars as mcal
import datetime as dt
import utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # Input and output paths
    trade_data_dir = os.path.join(SCRIPT_DIR, '../data/raw/etfs/market/tcbbo/exchange=XNAS.BASIC/')
    output_dir = os.path.join(SCRIPT_DIR, '../data/processed/etfs/market/etf_minute_volume_stats')
    
    # Load parameters
    params = utils.load_params()
    start_date = params['start_date']
    end_date = params['end_date']
    lookback_days = params['avg_dollar_volume_lookback_days']

    # getting data offset timedelta
    data_offset = pd.Timedelta(**params['data_offset'])
    
    # Get NYSE trading hours
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    # Convert to NY timezone
    schedule['market_open'] = schedule['market_open'].dt.tz_convert('America/New_York')
    schedule['market_close'] = schedule['market_close'].dt.tz_convert('America/New_York')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of tickers from directory structure
    import glob
    ticker_dirs = glob.glob(os.path.join(trade_data_dir, 'ticker=*'))
    tickers = [os.path.basename(d).replace('ticker=', '') for d in ticker_dirs]
    
    print(f"Processing {len(tickers)} tickers...")
    for ticker in tickers:
        print(f"Reading trade tick data for {ticker}...")
        
        # Read only this ticker's data
        ticker_df = pd.read_parquet(
            trade_data_dir,
            filters=[
                ('ticker', '=', ticker),
                ('date', '>=', start_date), 
                ('date', '<=', end_date)
            ],
            columns=['ticker', 'date', 'price', 'size']
        )
        
        if ticker_df.empty:
            print(f"No data for {ticker}, skipping...")
            continue
        
        print(f"Loaded {len(ticker_df)} trades for {ticker}")
        
        # Ensure index is timezone aware and convert to NY timezone
        if ticker_df.index.tz is None:
            ticker_df.index = ticker_df.index.tz_localize('UTC')
        ticker_df.index = ticker_df.index.tz_convert('America/New_York')
        
        # Filter for trading hours
        ticker_df['date'] = ticker_df.index.tz_localize(None).normalize()

        # Merge with schedule to get trading hours for each day
        ticker_df = ticker_df.merge(
            schedule[['market_open', 'market_close']], 
            left_on='date', 
            right_index=True,
            how='inner'
        )
        
        # Keep only trades during market hours
        ticker_df = ticker_df[
            (ticker_df.index >= ticker_df['market_open']) & 
            (ticker_df.index <= ticker_df['market_close'])
        ]
        
        if ticker_df.empty:
            print(f"No trading hour data for {ticker}, skipping...")
            continue
        
        # Calculate dollar volume for each trade
        ticker_df['dollar_volume'] = ticker_df['price'] * ticker_df['size']
        
        # Resample to 1-minute bars
        # Group by 1-minute bins
        minute_data = ticker_df.groupby(pd.Grouper(freq='1min', offset=data_offset)).agg({
            'dollar_volume': 'sum',
            'size': 'sum'
        })

        minute_data.index = minute_data.index - data_offset
        
        # Calculate VWAP
        minute_data['vwap'] = minute_data['dollar_volume'] / minute_data['size']
        minute_data['volume'] = minute_data['size']
        
        # Keep only rows with trades (volume > 0)
        minute_data = minute_data[minute_data['volume'] > 0]
        
        # Add date and time columns for grouping
        minute_data['date'] = minute_data.index.strftime('%Y-%m-%d')
        minute_data['time'] = minute_data.index.time
        
        # Calculate rolling average dollar volume by time of day across trading days
        minute_data['avg_dollar_volume'] = (
            minute_data.groupby('time')['dollar_volume']
            .rolling(window=lookback_days, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        # Keep only relevant columns
        result_df = minute_data[['vwap', 'volume', 'avg_dollar_volume','date']].copy()
    
        # Save as partitioned parquet
        ticker_output_path = os.path.join(output_dir, f'ticker={ticker}')
        os.makedirs(ticker_output_path, exist_ok=True)
        result_df.to_parquet(ticker_output_path,
                            partition_cols=['date'],)
        
        print(f"Processed {ticker}: {len(result_df)} minute bars")
    
    print(f"Saved minute VWAP data to {output_dir}")