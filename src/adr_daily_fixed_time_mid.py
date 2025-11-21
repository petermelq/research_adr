import datetime as dt
import pandas as pd
import numpy as np
from linux_xbbg import blp
import pandas_market_calendars as mcal
import glob

if __name__=='__main__':
    nbbo_dir = '/home/pmalonis/intraday_mean_reversion/data/raw/adrs/bbo-1m/nbbo'
    adr_info_filename = '/home/pmalonis/adr_trade/notebooks/brit_exchange_traded_efa_adrs_detailed.csv'

    # Fixed time of day to save mid price
    time_to_save = dt.time(13,0)

    # Read ADR info
    adr_info = pd.read_csv(adr_info_filename)
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity','')
    adr_tickers = adr_info['adr'].tolist()
    # Create close times dataframe
    start_date = '2020-01-02'
    end_date = '2025-11-30'
    ny_close_times = (
        mcal.get_calendar('NYSE').schedule(start_date=start_date,
                                            end_date=end_date)['market_close']
                                            .dt.tz_convert('America/New_York')
    )
    ny_close_times.index = ny_close_times.index.astype(str)
    start_time = (dt.datetime.combine(dt.date.today(), time_to_save) - pd.Timedelta('30min')).time()
    all_mid = {}
    for adr_ticker in adr_tickers:
        df = pd.read_parquet(nbbo_dir, filters=[('ticker', '==', adr_ticker)], columns=['nbbo_bid','nbbo_ask','date'])
        df['mid'] = (df['nbbo_bid'] + df['nbbo_ask']) / 2
        df = df.merge(ny_close_times, left_on='date', right_index=True)
        df = df[df['market_close'].dt.time == dt.time(16,0)]
        
        df = df.between_time(start_time, time_to_save)
        mid_df = df.groupby('date')['mid'].last()
        
        all_mid[adr_ticker] = mid_df
        print(f"Processed mid for {adr_ticker}")

    mid_df = pd.DataFrame(all_mid)
    mid_df.index = pd.to_datetime(mid_df.index)
    mid_df.to_csv(f'/home/pmalonis/adr_trade/data/processed/brit_adr_daily_mid_time={time_to_save}.csv')