import argparse
import datetime as dt
import pandas as pd
import numpy as np
from linux_xbbg import blp
import pandas_market_calendars as mcal
import glob
from . import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract daily fixed time mid prices for ADRs')
    parser.add_argument('--nbbo_dir',
                        type=str,
                        help='Directory containing NBBO parquet files'
                        )
    parser.add_argument("--tickers",
                        nargs="+",
                        type=str,
                        help="Tickers to download, or path to CSV containing a list of tickers."
                        )
    parser.add_argument('--time_to_save_hrs', 
                        type=int,
                        help='Hour of day (24h) to save mid price (NY time)'
                        )
    parser.add_argument('--time_to_save_mins', 
                        type=int,
                        default=0,
                        help='Minute of hour to save mid price (NY time)'
                        )
    parser.add_argument('--output_filename',
                        type=str,
                        default='/home/pmalonis/adr_trade/data/processed/daily_fixed_time_mid.csv',
                        help='Output filename for daily fixed time mid prices CSV'
                    )
    
    args = parser.parse_args()
    nbbo_dir = args.nbbo_dir
    adr_info_filename = '/home/pmalonis/adr_trade/notebooks/brit_exchange_traded_efa_adrs_detailed.csv'
    if args.tickers and len(args.tickers) == 1 and args.tickers[0].endswith(".csv"):
        tickers = pd.read_csv(args.tickers[0]).iloc[:,0].tolist()
    else:
        tickers = args.tickers

    # Fixed time of day to save mid price
    time_to_save = dt.time(args.time_to_save_hrs, args.time_to_save_mins)
    
    dates = utils.get_partition_dates(nbbo_dir)
    start_date = dates[0]
    end_date = dates[-1]
    # Create close times dataframe
    ny_close_times = (
        mcal.get_calendar('NYSE').schedule(start_date=start_date,
                                            end_date=end_date)['market_close']
                                            .dt.tz_convert('America/New_York')
    )
    ny_close_times.index = ny_close_times.index.astype(str)
    start_time = (dt.datetime.combine(dt.date.today(), time_to_save) - pd.Timedelta('30min')).time()
    all_mid = {}
    for adr_ticker in tickers:
        df = pd.read_parquet(nbbo_dir, filters=[('ticker', '==', adr_ticker)],
                            columns=['nbbo_bid','nbbo_ask','date'])
        df['mid'] = (df['nbbo_bid'] + df['nbbo_ask']) / 2
        df = df.merge(ny_close_times, left_on='date', right_index=True)
        df = df[df['market_close'].dt.time == dt.time(16,0)]
        
        df = df.between_time(start_time, time_to_save)
        mid_df = df.groupby('date')['mid'].last()
        
        all_mid[adr_ticker] = mid_df
        print(f"Processed mid for {adr_ticker}")

    mid_df = pd.DataFrame(all_mid)
    mid_df.index = pd.to_datetime(mid_df.index)
    mid_df.to_csv(args.output_filename)