import os
import numpy as np
import pandas as pd
from linux_xbbg import blp
import pandas_market_calendars as mcal
from .closing_domestic_prices import get_sh_per_adr
script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    adr_info_filename = os.path.join(script_dir, '..', 'data', 'raw', 'adr_info.csv')
    fx_dir = os.path.join(script_dir, '..', 'data', 'raw', 'currencies', 'minute_bars')

    for adjust in ['none', 'all']:
        raw_local_price_filename = os.path.join(script_dir, '..', 'data', 'raw', 'ordinary', f'ord_PX_OPEN_adjust_{adjust}.csv')
        output_filename = os.path.join(script_dir, '..', 'data', 'processed', 'ordinary', f'ord_open_to_usd_adr_PX_OPEN_adjust_{adjust}.csv')

        local_price_df = pd.read_csv(raw_local_price_filename, index_col=0, parse_dates=True).sort_index()
        start_date = local_price_df.index[0].strftime('%Y-%m-%d')
        end_date = local_price_df.index[-1].strftime('%Y-%m-%d')

        adr_info = pd.read_csv(adr_info_filename)
        tickers = adr_info['id'].tolist()

        exchanges = adr_info['exchange'].unique().tolist()
        open_time = pd.DataFrame({ex: mcal.get_calendar(ex).schedule(start_date=start_date, end_date=end_date)['market_open'].dt.tz_convert('America/New_York') for ex in exchanges})        
        open_time = open_time.stack().reset_index(name='open_time').rename(columns={'level_0':'date','level_1':'exchange'})

        currency_map = {'GBp': 'GBP'}
        needed_currencies = sorted(
            set(adr_info['currency'].dropna().unique().tolist())
            .intersection({'GBp', 'GBP', 'EUR', 'JPY', 'AUD', 'NOK', 'SEK', 'DKK', 'CHF', 'HKD'})
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
        stacked_price = pd.merge(stacked_price, adr_info[['id','exchange','currency']],
                                left_on='ticker', right_on='id', how='left').drop(columns=['id'])
        stacked_price = pd.merge(stacked_price, open_time, on=['date','exchange'], how='left')

        stacked_price = stacked_price[stacked_price['currency'].isin(needed_currencies)]

        stacked_price = pd.merge(stacked_price, fx_df[['timestamp','open','currency']].rename(columns={'open':'fx_rate'}), left_on=['open_time','currency'], right_on=['timestamp','currency'], how='left').drop(columns=['timestamp'])
        stacked_price['price_usd'] = stacked_price['price'] * stacked_price['fx_rate']
        stacked_price['adr_equivalent_price_usd'] = stacked_price['price_usd'] * stacked_price['sh_per_adr']

        price_df = stacked_price.pivot(index='date', columns='ticker', values='adr_equivalent_price_usd')
        output_dir = os.path.dirname(output_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        price_df.to_csv(output_filename)
