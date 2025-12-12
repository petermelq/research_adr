import os
import yaml
import pandas as pd
import datetime as dt
import pandas_market_calendars as mcal
import utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    # Input and output paths
    #trade_data_dir = os.path.join(SCRIPT_DIR,'..','data','raw','adrs','tcbbo','exchange=XNAS.BASIC')
    minute_bar_dir = os.path.join(SCRIPT_DIR,'..','data','processed','futures','converted_minute_bars')
    output_filename = os.path.join(SCRIPT_DIR,'..','data','processed','adrs','futures_volume_since_local_close.parquet')
    adr_info_filename = os.path.join(SCRIPT_DIR,'..','data','raw','adr_info.csv')

    adr_info = pd.read_csv(adr_info_filename)
    adr_tickers = adr_info['adr'].str.replace(' US Equity','').tolist()

    adr_info['adr_ticker'] = adr_info['adr'].str.replace(' US Equity','')
    exchange_dict = adr_info.set_index('adr_ticker')['exchange'].to_dict()
    futures_dict = adr_info.set_index('adr_ticker')['index_future_bbg']

    futures_symbols = pd.read_csv(os.path.join(SCRIPT_DIR, '..','data','raw','futures_symbols.csv'))
    futures_symbols = futures_symbols.set_index('bloomberg_symbol')['first_rate_symbol'].to_dict()

    params = utils.load_params()
    start_date = params['start_date']
    end_date = params['end_date']
    
    exchanges = adr_info['exchange'].unique().tolist()
    close_time = pd.DataFrame({ex: mcal.get_calendar(ex).schedule(start_date=start_date,
                                                                end_date=end_date)['market_close'].dt.tz_convert('America/New_York').rename('market_close') for ex in exchanges})
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
    if 'XOSL' in close_time.columns:
        close_time['XOSL'] += pd.Timedelta('5min')  # Oslo auction time 5 minutes after close
    if 'XSTO' in close_time.columns:
        close_time['XSTO'] += pd.Timedelta('0min')  # Stockholm auction time 0 minutes after close
    if 'XSWX' in close_time.columns:
        close_time['XSWX'] += pd.Timedelta('1min')  # Swiss auction time 1 minute after close
    if 'XCSE' in close_time.columns:
        close_time['XCSE'] += pd.Timedelta('0min')  # CSE auction time 0 minutes after close

    ny_close = mcal.get_calendar('NYSE').schedule(start_date=start_date,
                                                  end_date=end_date)['market_close'].dt.tz_convert('America/New_York')
    all_ticker_data = []
    for ticker in adr_tickers:
        futures_symbol = futures_symbols[futures_dict[ticker]]
        df = pd.read_parquet(minute_bar_dir,
                             filters=[('symbol','==', futures_symbol)])
        df['date'] = df['timestamp'].dt.tz_localize(None).dt.normalize()
        df = df[(df['date'] >= start_date) &
                (df['date'] <= end_date)]
        df['date'] = pd.to_datetime(df['date'].astype(str))
        df = df.set_index('timestamp')
        exchange = exchange_dict.get(ticker)
        if exchange is None:
            print(f"No exchange mapping for {ticker}, skipping...")
            continue
        
        try:
            df = df.merge(close_time[exchange].rename('market_close'), left_on='date', right_index=True)
            df = df.merge(ny_close.rename('ny_market_close_time'), left_on='date', right_index=True)
            df = df[(df.index >= df['market_close']) &
                    (df.index <= df['ny_market_close_time'])]
            df['cumulative_volume'] = df.groupby('date')['volume'].transform('cumsum')
        except Exception as e:
            import IPython; IPython.embed()

        all_ticker_data.append(df['cumulative_volume'].rename(ticker))
        print(f"Processed {ticker}")
        
    print(f"Saving output to {output_filename}")
    pd.concat(all_ticker_data, axis=1).to_parquet(output_filename)