import os
import polars as pl
import pandas as pd
import pandas_market_calendars as mcal
from linux_xbbg import blp
from pathlib import Path
from utils import get_market_business_days, load_params
__script_dir__ = os.path.dirname(os.path.abspath(__file__))

def process_adr_mids_efficiently(adr_path, ticker_close_df, start_date=None, end_date=None):
    """
    Memory-efficient processing of partitioned ADR data using Polars.
    Processes data in chunks by ticker partitions. Handles various partitioning schemes:
    - ticker/date partitions (date in partition path)
    - ticker/year/month partitions (date in parquet file)
    - ticker/year partitions (date in parquet file)
    - ticker only (date in parquet file)
    """
    # Convert ticker_close to polars - avoid pandas conversion by recreating
    ticker_close_pl = pl.DataFrame({
        'adr': ticker_close_df['adr'].tolist(),
        'exchange': ticker_close_df['exchange'].tolist(),
        'date': [pd.to_datetime(d) for d in ticker_close_df['date'].tolist()],
        'close_time': [pd.to_datetime(ct) for ct in ticker_close_df['close_time'].tolist()]
    })
    
    # Get all partition directories
    adr_path = Path(adr_path)
    ticker_dirs = [d for d in adr_path.iterdir() if d.is_dir() and d.name.startswith('ticker=')]
    
    results = []
    processed_count = 0
    
    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name.split('=')[1]
        
        # Filter ticker_close for this specific ticker
        ticker_close_filtered = ticker_close_pl.filter(pl.col('adr') == ticker)
        
        if ticker_close_filtered.height == 0:
            continue  # Skip if no close times for this ticker
        
        # Find all parquet files recursively under this ticker directory
        parquet_files = list(ticker_dir.rglob('*.parquet'))
        
        if not parquet_files:
            continue
        
        for parquet_file in parquet_files:
            try:
                # Read single partition efficiently with Polars
                adr_data = pl.read_parquet(parquet_file)
                
                # Check if 'date' column exists in the data
                if 'date' not in adr_data.columns:
                    print(f"Warning: 'date' column not found in {parquet_file}, skipping")
                    continue
                
                # Ensure date column is proper datetime type
                # The date column might be categorical or string, so cast to string first then parse
                adr_data = adr_data.with_columns([
                    pl.col('date').cast(pl.Utf8).str.to_date().alias('date')
                ])
                
                # Apply date filtering if specified
                if start_date:
                    start_dt = pd.to_datetime(start_date).date()
                    adr_data = adr_data.filter(pl.col('date') >= start_dt)
                
                if end_date:
                    end_dt = pd.to_datetime(end_date).date()
                    adr_data = adr_data.filter(pl.col('date') <= end_dt)
                
                if adr_data.height == 0:
                    continue
                
                # Create mid column
                adr_data = adr_data.with_columns([
                    ((pl.col('nbbo_bid') + pl.col('nbbo_ask')) / 2).alias('mid')
                ])
                
                # Select only necessary columns to reduce memory usage
                # Use the Ticker column from the data itself
                adr_data = adr_data.select(['ts_recv', 'mid', 'Ticker', 'date'])
                
                # Get unique dates in this partition
                unique_dates = adr_data.select('date').unique().to_series().to_list()
                
                for partition_date in unique_dates:
                    # Filter data for this specific date
                    date_data = adr_data.filter(pl.col('date') == partition_date)
                    
                    if date_data.height == 0:
                        continue
                    
                    # Filter ticker_close for this specific date
                    date_close_filtered = ticker_close_filtered.filter(
                        pl.col('date').dt.date() == partition_date
                    )
                    
                    if date_close_filtered.height == 0:
                        continue  # Skip if no close time for this date
                    
                    # Convert close time to same timezone and precision as ts_recv for proper joining
                    # Note: ts_recv uses 'America/New_York' timezone
                    date_close_converted = date_close_filtered.with_columns([
                        pl.col('close_time').dt.convert_time_zone('America/New_York').dt.cast_time_unit('ns')
                    ])
                    
                    # Join with ticker_close for this specific ticker/date combination
                    # Use asof join to find the closest mid price to each close time
                    # No need to join by ticker since we're already filtering by ticker in the outer loop
                    joined_data = date_close_converted.join_asof(
                        date_data.sort('ts_recv'),
                        left_on='close_time',
                        right_on='ts_recv'
                    )
                    
                    if joined_data.height > 0:
                        results.append(joined_data)
                        processed_count += joined_data.height
                        print(f"Processed {ticker} for {partition_date}: {joined_data.height} records")
                    
            except Exception as e:
                print(f"Error processing {ticker} in {parquet_file}: {e}")
                continue
    
    print(f"Total processed records: {processed_count}")
    
    # Combine all results
    if results:
        final_result = pl.concat(results)
        return final_result
    else:
        return pl.DataFrame()

cbday = get_market_business_days()
def get_daily_adj(adj_df, start_date, end_date):
    adj_df = adj_df.groupby('adjustment_date')[['adjustment_factor']].prod().sort_index(ascending=False)
    adj_df.loc[start_date, 'adjustment_factor'] = 1.0
    adj_df['cum_adj'] = adj_df['adjustment_factor'].cumprod()
    adj_df.index = [pd.to_datetime(idx) - cbday for idx in adj_df.index]
    adj_df.loc[end_date, 'cum_adj'] = 1.0
    adj_df = adj_df.sort_index().loc[:end_date]
    adj_df = adj_df[['cum_adj']].sort_index().resample('1D').bfill()
    
    return adj_df

if __name__ == '__main__':
    adr_path = os.path.join(__script_dir__, '..', 'data', 'raw', 'adrs', 'bbo-1m', 'nbbo')
    adr_info_filename = os.path.join(__script_dir__, '..', 'data', 'raw', 'adr_info.csv')

    params = load_params()
    start_date = params['start_date']
    end_date = params['end_date']
        
    # Read holdings data with pandas (small dataset)
    adr_info = pd.read_csv(adr_info_filename)
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity','')
    exchanges = adr_info['exchange'].unique().tolist()
    
    # Create close times dataframe
    close_time = pd.DataFrame({ex: mcal.get_calendar(ex).schedule(start_date=start_date, end_date=end_date)['market_close'].dt.tz_convert('America/New_York') for ex in exchanges})
    if 'XLON' in close_time.columns:
        close_time['XLON'] += pd.Timedelta('6min')  # London auction time 6 minutes after close
    if 'XAMS' in close_time.columns:
        close_time['XAMS'] += pd.Timedelta('6min')  # Amsterdam auction time 6 minutes after close
    if 'XPAR' in close_time.columns:
        close_time['XPAR'] += pd.Timedelta('6min')  # Paris auction time 6 minutes after close
    if 'XETR' in close_time.columns:
        close_time['XETR'] += pd.Timedelta('6min')  # Frankfurt auction time 6 minutes after close
    if 'XMIL' in close_time.columns:
        close_time['XMIL'] += pd.Timedelta('6min')  # Milan auction time 6 minutes after close
    if 'XBRU' in close_time.columns:
        close_time['XBRU'] += pd.Timedelta('6min')  # Brussels auction time 6 minutes after close
    if 'XMAD' in close_time.columns:
        close_time['XMAD'] += pd.Timedelta('6min')  # Madrid auction time 6 minutes after close
    if 'XHEL' in close_time.columns:
        close_time['XHEL'] += pd.Timedelta('0min')  # Helsinki auction time 0 minutes after close
    if 'XDUB' in close_time.columns:
        close_time['XDUB'] += pd.Timedelta('0min')  # Dublin auction time 0 minutes after close

    close_time = close_time.stack().reset_index(name='close_time').rename(columns={'level_0':'date','level_1':'exchange'})
    ticker_close = adr_info[['adr','exchange']].merge(close_time, on='exchange')
    
    # Process ADR data efficiently
    print("Processing ADR mids at ordinary close times...")
    result_df = process_adr_mids_efficiently(adr_path, 
                                            ticker_close,
                                            start_date=start_date,
                                            end_date=end_date)
    
    print('result_df:', result_df)
    result_df = result_df.pivot(on='adr',index='date',values='mid').to_pandas().set_index('date').sort_index()

    # adjusting prices for splits/dividends
    adjustment_filename = os.path.join(__script_dir__, '..', 'data', 'processed', 'adrs', 'adr_adjustment_factors.csv')
    adj_factors = pd.read_csv(adjustment_filename)
    adj_df = (adj_factors.groupby('ticker')
                .apply(get_daily_adj, start_date=start_date, end_date=end_date)
                .reset_index().rename(columns={'level_1':'date'}))
    
    stacked_price = result_df.stack().reset_index(name='price').rename(columns={'level_1':'ticker'})
    adj_df['ticker'] = adj_df['ticker'].str.replace(' US Equity','')
    merged = stacked_price.merge(adj_df, on=['ticker','date'], how='left')
    merged['adj_price'] = merged['price'] * merged['cum_adj']
    adj_result_df = merged.pivot(index='date', columns='ticker', values='adj_price')

    # Save results
    if result_df.shape[0] > 0:
        output_file = os.path.join(__script_dir__, 
                                   '..', 
                                   'data', 
                                   'processed', 
                                   'adrs',
                                   'adr_mid_at_ord_auction_adjust_none.csv'
                                )
        result_df.to_csv(output_file)
        adj_output_file = os.path.join(__script_dir__,
                                       '..',
                                       'data',
                                       'processed',
                                       'adrs',
                                       'adr_mid_at_ord_auction_adjust_all.csv'
                                    )
        adj_result_df.to_csv(adj_output_file)
        print(f"Results saved to {output_file}")
        print(f"Processed {result_df.shape[0]} records")
    else:
        print("No data processed - check file paths and data availability")