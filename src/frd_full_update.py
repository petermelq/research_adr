import string
import ray.remote_function
import yaml
from arcticdb import Arctic
import io
import zipfile
import pandas as pd
import requests
import datetime as dt
import os
import argparse
import utils
import ray
from glob import glob
import sys
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)
import utils

url = "https://firstratedata.com/api/data_file"

def get_period(output_dir, adjustment):
    date_strs = sorted([d[-10:] for d in glob(os.path.join(output_dir, 'date=*'))])
    if len(date_strs) == 0 or adjustment != 'UNADJUSTED':
        period = 'full'
        last_date = None
    else:
        last_date = pd.Timestamp(date_strs[-1])
        today_tstamp = pd.Timestamp(dt.datetime.now().date())
        n_missing_days = utils.trading_day_diff(last_date, today_tstamp)
        if n_missing_days == 0:
            period = None
        elif n_missing_days == 1:
            period = 'day'
        elif 1 < n_missing_days <= 5:
            period = 'week'
        elif 5 < n_missing_days <= 20:
            period = 'month'
        else:
            period = 'full'

    return period, last_date

def save_data(output_dir, url, params, start_date, trade_tickers):
    print('Request sent with parameters: \n', params)
    response = requests.get(url, params=params)
    try:
        z = zipfile.ZipFile(io.BytesIO(response.content))
    except:
        import IPython; IPython.embed()

    for ticker_file in z.infolist():
        ticker = ticker_file.filename.split('_')[0]
        if ticker not in trade_tickers:
            continue
        else:
            try:
                df = pd.read_csv(io.StringIO(z.read(ticker_file).decode('utf-8')), 
                            names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            except Exception as e:
                print(f'Formatting issue for ticker {ticker}')
                raise e
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            except:
                import pdb;pdb.set_trace()

            df.dropna(subset=['DateTime'], inplace=True)
            df.set_index('DateTime', inplace=True)
            df = df.sort_index(ascending=True)
            df = df.loc[start_date:]
            df['Volume'] = df['Volume'].astype(float)

            ticker = ticker_file.filename.split('_')[0]
            for date, date_df in df.groupby(df.index.date):
                date_str = date.strftime('%Y-%m-%d')
                dirname = os.path.join(output_dir, f'date={date_str}', f'ticker={ticker}')
                os.makedirs(dirname, exist_ok=True)
                date_df.to_parquet(f'{dirname}/data.parquet')

            print(f"Saved {ticker}")

if __name__ == '__main__':
    # adding an argparser to get the adjustment type as a store_true
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root',
                        required=False,
                        default='minute_bars',
                        type=str,
                        help='Root of output filename. Adjustment type is added to the root',
                        )
    parser.add_argument('--adjusted',
                        action='store_true',
                        help='Download adjusted data',
                        )
    parser.add_argument('--timeframe',
                        required=False,
                        default='1min',
                        type=str,
                        help='Timeframe for data',
                        )
    args = parser.parse_args()

    if args.adjusted:
        adjustment = 'adj_splitdiv'
    else:
        adjustment = 'UNADJUSTED'

    data_dir = os.path.join(script_dir, '..', 'data', 'raw')
    output_dirname = f'{args.output_root}_{adjustment}'
    output_path = os.path.join(data_dir, output_dirname)
    print('Saving to', output_path)
    period, last_date = get_period(output_path, adjustment)
    timeframe = args.timeframe

    start_date = pd.Timestamp(yaml.safe_load(open(os.path.join(script_dir, '..', 'params.yaml')))['start_date'])
    trade_tickers = pd.read_csv(os.path.join(script_dir, '..', 'data', 'raw', 'adr_info.csv'))['adr'].str.replace(' US Equity','').tolist()
    
    cbday = utils.get_market_business_days('XNYS')

    letters = string.ascii_uppercase
    params = {
                "period": period,
                "timeframe": f"{timeframe}",
                "adjustment": adjustment,
                "userid": "fjm3nF2GrUe7fK8dhmcx6A",
            }
    
    for asset_type in ['stock']:
        params.update({"type": asset_type})
        if period == 'full':
            ids = []
            for ticker_range in letters:
                params.update({"ticker_range": ticker_range})
                save_data(output_path, url, params, start_date, trade_tickers)

        else:
            save_data(output_path, url, params, start_date, trade_tickers)