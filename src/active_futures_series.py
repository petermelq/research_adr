import re
import os
import argparse
import numpy as np
import pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

futures_symbols = pd.read_csv(os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures_symbols.csv'))
futures_map = futures_symbols.set_index('exchange_symbol')['bloomberg_symbol'].to_dict()
def eurex_to_bbg(symbol):
    MONTH_CODES = 'FGHJKMNQUVXZ'
    month_map = {
        'JAN': 'F',
        'FEB': 'G',
        'MAR': 'H',
        'APR': 'J',
        'MAY': 'K',
        'JUN': 'M',
        'JUL': 'N',
        'AUG': 'Q',
        'SEP': 'U',
        'OCT': 'V',
        'NOV': 'X',
        'DEC': 'Z',
    }
    s = symbol.split()
    if len(s) == 1: # spread
        instrument,_,front_code,back_code,_ = symbol.split('.')
        front_month = month_map[front_code[:3]]
        front_year = front_code[3:4]
        back_month = month_map[back_code[:3]]
        back_year = back_code[3:4]

        if instrument in futures_map:
            bbg_instrument = futures_map[instrument]
        else:
            raise ValueError(f'Unrecognized ICE instrument: {instrument}')

        bbg = (bbg_instrument + front_month + front_year +
               bbg_instrument + back_month + back_year + ' Index')
        
    elif len(s) == 4: # outright
        instrument,_,maturity,_ = symbol.split()
        try:
            month = MONTH_CODES[int(maturity[4:6])-1]
        except:
            import IPython; IPython.embed()
        year = maturity[2:4]

        if instrument in futures_map:
            bbg_instrument = futures_map[instrument]
        else:
            raise ValueError(f'Unrecognized ICE instrument: {instrument}')

        bbg = bbg_instrument + month + year + ' Index'
    else:
        raise ValueError(f'Unrecognized Eurex instrument format: {symbol}')

    return bbg

def parse_ice_maturity(maturity_code):
    month = maturity_code[2]
    year = re.split(r'[-_!]', maturity_code, maxsplit=1)[0][-2:]

    return month, year

def ice_to_bbg(symbol):
    s = symbol.split()
    instrument = s[0]
    if instrument in futures_map:
        bbg_instrument = futures_map[instrument]
    else:
        raise ValueError(f'Unrecognized ICE instrument: {instrument}')
    
    if len(s) == 2: # outright
        maturity_code = s[1]
        month, year = parse_ice_maturity(maturity_code)
        bbg_symbol = bbg_instrument + month + year + ' Index'
    elif len(s) == 3: # spread
        front_code, back_code = s[1:]
        front_month, front_year = parse_ice_maturity(front_code)
        back_month, back_year = parse_ice_maturity(back_code)
        bbg_symbol = (bbg_instrument + front_month + front_year + ' '
                   + bbg_instrument + back_month + back_year + ' Index')
    else:
        raise ValueError(f'Unrecognized ICE instrument format: {instrument}')

    return bbg_symbol

def year_digit_to_two_digit(year_digit):
    year_int = int(year_digit)
    if year_int < 7:
        return '2' + year_digit
    else:
        return '1' + year_digit

def cme_to_bbg(symbol):
    s = symbol.split('-')
    if len(s) == 1: # outright
        year = year_digit_to_two_digit(s[0][-1])
        month = s[0][-2]
        instrument = s[0][:-2]

        if instrument in futures_map:
            bbg_instrument = futures_map[instrument]
        else:
            raise ValueError(f'Unrecognized CME instrument: {instrument}')

        bbg_symbol = bbg_instrument + month + year + ' Index'

    elif len(s) == 2: # spread
        front_symbol, back_symbol = s
        front_year = year_digit_to_two_digit(front_symbol[-1])
        front_month = front_symbol[-2]
        front_instrument = front_symbol[:-2]
        back_year = year_digit_to_two_digit(back_symbol[-1])
        back_month = back_symbol[-2]
        back_instrument = back_symbol[:-2]

        bbg_front_instrument = futures_map[front_instrument]
        bbg_back_instrument = futures_map[back_instrument]
        
        bbg_symbol = (bbg_front_instrument + front_month + front_year + 
                        bbg_back_instrument + back_month + back_year + ' Index')
    else:
        raise ValueError(f'Unrecognized CME instrument format: {symbol}')
                        
    return bbg_symbol

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process active futures series.')
    parser.add_argument('fut_dir', type=str, help='Directory containing futures minute bar data.')
    parser.add_argument('output_dir', type=str, help='Directory to save active futures series.')
    args = parser.parse_args()
    fut_dir = args.fut_dir
    output_dir = args.output_dir

    futures_symbol_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures_symbols.csv')
    futures_volume_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'daily_futures_volume.parquet')
    adr_info_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    adr_info = pd.read_csv(adr_info_filename)
    futures_symbols = pd.read_csv(futures_symbol_filename)

    bbg_futures = adr_info['index_future_bbg'].unique().tolist()
    exchange_futures = [futures_symbols[futures_symbols['bloomberg_symbol']==fut]['exchange_symbol'].iloc[0] for fut in bbg_futures]

    volume_df = pd.read_parquet(futures_volume_filename)
    active_contracts = (volume_df.T.groupby(level=0)
                        .apply(lambda x: x.dropna(how='all',axis=1).idxmax(skipna=True))
                        .unstack().T
                        .map(lambda x: x[1]
                                        if isinstance(x,tuple)
                                        else np.nan)
                        .ffill()
                    )
    
    for bbg_symbol, exchange_symbol in zip(bbg_futures, exchange_futures):
        futures_df = pd.read_parquet(fut_dir,
                                    filters=[('code','==',exchange_symbol)])
        futures_df['date'] = futures_df['date'].astype('datetime64[ns]')
        exchanges = futures_df['exchange'].unique().tolist()
        assert len(exchanges) == 1, f'Multiple exchanges found for {exchange_symbol}: {exchanges}'
        exchange = exchanges[0]

        if exchange == 'XEUR.EOBI':
            bbg_map = eurex_to_bbg  
        elif exchange == 'GLBX.MDP3':
            bbg_map = cme_to_bbg
        elif exchange == 'IFLL.IMPACT':
            bbg_map = ice_to_bbg

        try:
            futures_df['bbg_symbol'] = futures_df['symbol'].apply(bbg_map)
        except Exception as e:
            print(f'Error processing {bbg_symbol} ({exchange_symbol}) with exchange {exchange}')
            raise e
            
        futures_df = futures_df.merge(active_contracts[bbg_symbol].rename('active_bbg'),
                                        left_on='date', right_index=True, how='left')
        futures_df = futures_df[futures_df['bbg_symbol'] == futures_df['active_bbg']]
        output_path = os.path.join(output_dir, f'exchange={exchange}', f'code={exchange_symbol}')
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)

        os.makedirs(output_path, exist_ok=True)
        futures_df['date'] = futures_df['date'].dt.strftime('%Y-%m-%d')
        futures_df.to_parquet(output_path, partition_cols=['date'])
        
        print(f'Processed {bbg_symbol} ({exchange_symbol}) with exchange {exchange}, saved to {output_path}')