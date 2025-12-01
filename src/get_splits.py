import os
import pandas as pd
from linux_xbbg import blp
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    adr_info_path = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    sector_etf_path = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'sector_etfs.csv')
    adr_info = pd.read_csv(adr_info_path)
    sector_etfs = pd.read_csv(sector_etf_path)
    all_tickers = (list(adr_info['id'].unique()) + 
                   list(adr_info['adr'].unique()) + 
                   list((adr_info['market_etf_hedge'] + ' US Equity').unique()) +
                   list((sector_etfs['hedge'].dropna() + ' US Equity').unique())
                )
    
    split_df = blp.bds(all_tickers,
                    'EQY_DVD_ADJUST_FACT',
                    Corporate_Actions_Filter='CAPITAL_CHANGE',)
    split_df.to_csv(os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'all_splits.csv'))