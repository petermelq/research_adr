import argparse
import numpy as np
import pandas as pd
import datetime as dt
import pandas_market_calendars as mcal
import utils

if __name__ == '__main__':
    domestic_close_mid = pd.read_csv('/home/pmalonis/adr_trade/data/processed/adr_mids_at_underlying_auction_adjust_none.csv')
    afternoon_mid_df = pd.read_csv(f'/home/pmalonis/adr_trade/data/processed/adr_daily_mid_time.csv')

    # Fixed time of day to save mid price
    params = utils.load_params()
    time_to_save = dt.time(params['time_to_save_hrs'], 
                           params['time_to_save_mins'])

    df = pd.read_parquet('../data/processed/FTUK_close_to_usd_1min.parquet')
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df = df.set_index('timestamp')
    
    time_futures_after_close = pd.Timedelta(minutes=5)
    adr_info_filename = '/home/pmalonis/adr_trade/notebooks/brit_exchange_traded_efa_adrs_detailed.csv'
    betas = pd.read_csv('/home/pmalonis/adr_trade/data/processed/brit_underlying_betas_to_UKX.csv', index_col=0)

    # Fixed time of day to compute signal
    time_to_save = dt.time(13,0)
    start_time = (dt.datetime.combine(dt.date.today(), time_to_save) - pd.Timedelta('30min')).time()

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

    futures_df = df.merge(ny_close_times.rename('ny_market_close'), left_on='date', right_index=True)
    futures_df = futures_df[futures_df['ny_market_close'].dt.time == dt.time(16,0)]
    futures_df = futures_df[futures_df['date'] > start_date]
    import IPython; IPython.embed();
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
                            )
        close_times[ex].index = close_times[ex].index.astype(str)
    
    all_signal = {}
    # def get_afternoon(x):
    #     try:
    #         return x[x.index <= x['market_close'] + time_futures_after_close].iloc[-1]['close']
    #     except:
    #         import IPython; IPython.embed();
    #         return np.nan

    for ticker in adr_tickers:
        exchange = exchange_dict[ticker]
        close_df = close_times[exchange]
        merged_fut = futures_df.merge(close_df, left_on='date', right_index=True)
        fut_domestic_close = merged_fut.groupby('date').apply(
            lambda x: x[x.index <= x['market_close'] + time_futures_after_close].iloc[-1]['close']
        )
        fut_afternoon = merged_fut.groupby('date').apply(
            lambda x: x[x.index.time <= time_to_save].iloc[-1]['close']
        )
        fut_ret = ((fut_afternoon - fut_domestic_close) / fut_domestic_close).to_frame(name='fut_ret')

        merged = fut_ret.merge(betas[ticker].rename('beta'), 
                               left_on='date', right_index=True)
        merged['signal'] = merged['fut_ret'] * merged['beta']

        all_signal[ticker] = merged['signal']

        print(f"Processed signal for {ticker}")

    all_signal_df = pd.DataFrame(all_signal)
    all_signal_df.to_csv(f'/home/pmalonis/adr_trade/data/processed/fixed_time_signal_time={time_to_save}.csv')