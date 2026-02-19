import os
import numpy as np
import pandas as pd
from linux_xbbg import blp
import pandas_market_calendars as mcal
from . import utils
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_sh_per_adr(start_date, end_date, for_adjusted):
    adr_info_path = os.path.join(script_dir, '..', 'data', 'raw', 'adr_info.csv')
    adr_info = pd.read_csv(adr_info_path)
    cbday = utils.get_market_business_days('NYSE')
    idx = pd.date_range(start=start_date, end=end_date, freq=cbday)
    sh_per_adr = pd.DataFrame({k:[v]*len(idx) for k,v in adr_info.set_index('adr')['sh_per_adr'].to_dict().items()},index=idx)

    if for_adjusted:
        reclass_path = os.path.join(script_dir, '..', 'data', 'raw', 'adrs', 'share_reclass.csv')
        reclass_df = pd.read_csv(reclass_path)
        reclass_df['Effective Date'] = pd.to_datetime(reclass_df['Effective Date'])
        reclass_df = reclass_df[(reclass_df['Effective Date'] >= pd.to_datetime(start_date)) &
                                (reclass_df['Effective Date'] <= pd.to_datetime(end_date))].sort_values('Effective Date', ascending=False)

        for _,row in reclass_df.iterrows():
            ticker = row['Security ID']
            old_ratio = row['Old_Ratio']
            sh_per_adr.loc[sh_per_adr.index < row['Effective Date'], ticker] = old_ratio
    else:
        split_path = os.path.join(script_dir, '..', 'data', 'raw', 'all_splits.csv')
        split_df = pd.read_csv(split_path, index_col=0, parse_dates=['ex_date']).sort_values('ex_date', ascending=False)
        split_df = split_df[(split_df['ex_date'] >= pd.to_datetime(start_date)) & (split_df['ex_date'] <= pd.to_datetime(end_date))]
        for ticker,row in split_df.iterrows():
            ratio = row['dvd_amt']
            if ticker in adr_info['adr'].values: # adr split
                sh_per_adr.loc[sh_per_adr.index < row['ex_date'], ticker] = sh_per_adr.loc[sh_per_adr.index < row['ex_date'], ticker] * ratio
            elif ticker in adr_info['id'].values: # ordinary split
                adr_ticker = adr_info[adr_info['id'] == ticker]['adr'].values[0]
                sh_per_adr.loc[sh_per_adr.index < row['ex_date'], adr_ticker] = sh_per_adr.loc[sh_per_adr.index < row['ex_date'], adr_ticker] / ratio

    sh_per_adr.columns = [col.replace(' US Equity','') for col in sh_per_adr.columns]

    return sh_per_adr

if __name__ == "__main__":
    adr_info_filename = os.path.join(script_dir, '..', 'data', 'raw', 'adr_info.csv')
    fx_dir = os.path.join(script_dir, '..', 'data', 'raw', 'currencies', 'minute_bars')

    for adjust in ['none', 'all']:
        raw_local_price_filename = os.path.join(script_dir, '..', 'data', 'raw', 'ordinary', f'ord_PX_LAST_adjust_{adjust}.csv')
        output_filename = os.path.join(script_dir, '..', 'data', 'processed', 'ordinary', f'ord_close_to_usd_adr_PX_LAST_adjust_{adjust}.csv')

        local_price_df = pd.read_csv(raw_local_price_filename, index_col=0, parse_dates=True).sort_index()
        start_date = local_price_df.index[0].strftime('%Y-%m-%d')
        end_date = local_price_df.index[-1].strftime('%Y-%m-%d')

        adr_info = pd.read_csv(adr_info_filename)
        tickers = adr_info['id'].tolist()

        exchanges = adr_info['exchange'].unique().tolist()
        close_time = pd.DataFrame({ex: mcal.get_calendar(ex).schedule(start_date=start_date, end_date=end_date)['market_close'].dt.tz_convert('America/New_York') for ex in exchanges})
        close_time = close_time.stack().reset_index(name='close_time').rename(columns={'level_0':'date','level_1':'exchange'})

        currency_map = {'GBp': 'GBP'}
        needed_currencies = sorted(
            set(adr_info['currency'].dropna().unique().tolist())
            .intersection({'GBp', 'GBP', 'EUR', 'JPY', 'AUD', 'NOK', 'SEK', 'DKK', 'CHF'})
        )

        all_fx_data = []
        for currency in needed_currencies:
            source_cur = currency_map.get(currency, currency)
            fx_file = os.path.join(fx_dir, f'{source_cur}USD_full_1min.txt')
            fx_df = pd.read_csv(fx_file, header=None, index_col=None, names=['date','time','open','high','low','close','volume'])
            fx_df['timestamp'] = pd.to_datetime(fx_df['date'].astype(str) + ' ' + fx_df['time'].astype(str)).dt.tz_localize('America/New_York')
            if currency == 'GBp':
                fx_df[['open','high','low','close']] = fx_df[['open','high','low','close']] / 100 # convert to GBpUSD
            fx_df['currency'] = currency
                
            all_fx_data.append(fx_df)
        
        fx_df = pd.concat(all_fx_data, ignore_index=True)

        stacked_price = local_price_df.stack().reset_index(name='price').rename(columns={'level_0':'date','level_1':'ticker'})
        if adjust == 'none':
            for_adjusted = False
        else:
            for_adjusted = True
        
        sh_per_adr = get_sh_per_adr(start_date, end_date, for_adjusted=for_adjusted)
        sh_per_adr = sh_per_adr.rename(columns=adr_info.set_index(adr_info['adr'].str.replace(' US Equity',''))['id'].to_dict())
        sh_per_adr = sh_per_adr.stack().to_frame(name='sh_per_adr').reset_index(names=['date','ticker'])
        
        stacked_price = pd.merge(stacked_price, sh_per_adr, on=['date','ticker'], how='left')
        stacked_price = pd.merge(stacked_price, adr_info[['id','exchange','currency']], left_on='ticker', right_on='id', how='left').drop(columns=['id'])
        stacked_price = pd.merge(stacked_price, close_time, on=['date','exchange'], how='left')

        stacked_price = stacked_price[stacked_price['currency'].isin(needed_currencies)]

        stacked_price = pd.merge(stacked_price, fx_df[['timestamp','close','currency']].rename(columns={'close':'fx_rate'}), left_on=['close_time','currency'], right_on=['timestamp','currency'], how='left').drop(columns=['timestamp'])
        stacked_price['price_usd'] = stacked_price['price'] * stacked_price['fx_rate']
        stacked_price['adr_equivalent_price_usd'] = stacked_price['price_usd'] * stacked_price['sh_per_adr']

        price_df = stacked_price.pivot(index='date', columns='ticker', values='adr_equivalent_price_usd')
        output_dir = os.path.dirname(output_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        price_df.to_csv(output_filename)
