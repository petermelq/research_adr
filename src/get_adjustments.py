import os
import sys
import argparse
import numpy as np
import pandas as pd
from linux_xbbg import blp
sys.path.append('../src')
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get adjustment factors from dividends and corporate actions. For ADRs this will reflect the net dividend rather than the gross dividend.')
    parser.add_argument('price_filename',
                        type=str,
                        help='Path to unadjusted price CSV file')
    parser.add_argument('output_filename',
                        type=str,
                        help='Output CSV filename for adjustment factors')
    parser.add_argument('--symbol_suffix',
                        type=str,
                        default='',
                        help='Suffix to append to each ticker in column of price file for Bloomberg')
    args = parser.parse_args()

    price_filename = args.price_filename
    output_filename = args.output_filename

    price_df = pd.read_csv(price_filename, index_col=0, parse_dates=True).sort_index()
    start_date = price_df.index[0].strftime('%Y-%m-%d')
    end_date = price_df.index[-1].strftime('%Y-%m-%d')

    tickers = price_df.columns.tolist()
    bbg_tickers = [t + args.symbol_suffix for t in tickers]
    div_df = blp.dividend(bbg_tickers,
                            start_date=start_date,
                            end_date=end_date,
                            typ='all',
                            timeout=5000,
                        ).reset_index(names='ticker')

    div_df.loc[(div_df['dvd_type'] == 'Cancelled'), 'dvd_amt'] = 0.0

    # creating adjustment for GSK spin-off
    if 'GSK US Equity' in bbg_tickers:
        div_df.loc[(div_df['ticker'] == 'GSK US Equity') & (div_df['dvd_type'] != 'Stock Split') & (div_df['ex_date'] == '2022-07-19'), 'dvd_amt'] = blp.bdh(['HLN US Equity'], flds=['PX_LAST'], start_date='2022-07-18', end_date='2022-07-18').iloc[0,0]
    
    cbday = utils.get_market_business_days('NYSE')
    div_df['last_cum_div'] = [d - cbday for d in pd.to_datetime(div_df['ex_date'])]

    stacked_close = price_df.stack().reset_index()
    stacked_close.columns = ['date','ticker','close_price']
    stacked_close['ticker'] = stacked_close['ticker'] + args.symbol_suffix
    merged_df = pd.merge(div_df,
                        stacked_close,
                        left_on=['ticker','last_cum_div'],
                        right_on=['ticker','date'],
                        how='left',
                    )
    
    dvd_idx = ~merged_df['dvd_type'].isin(['Stock Split', 'Rights Issue'])
    merged_df['adjustment_factor'] = np.nan
    merged_df.loc[dvd_idx, 'adjustment_factor'] = 1 - (merged_df.loc[dvd_idx, 'dvd_amt'] / merged_df.loc[dvd_idx, 'close_price'])

    split_idx = merged_df['dvd_type'] == 'Stock Split'
    merged_df.loc[split_idx, 'adjustment_factor'] = 1 / merged_df.loc[split_idx, 'dvd_amt']

    rights_idx = merged_df['dvd_type'] == 'Rights Issue'
    if rights_idx.sum() > 0:
        rights_tickers = merged_df.loc[rights_idx, 'ticker'].unique().tolist()
        adj_factors = blp.bds(rights_tickers,
                                'EQY_DVD_ADJUST_FACT',
                                Corporate_Actions_Filter='NORMAL_CASH|ABNORMAL_CASH|CAPITAL_CHANGE',)
        adj_factors = adj_factors.reset_index(names=['ticker'])

        rights_adj = adj_factors[(adj_factors['adjustment_factor_operator_type'] == 2.0) &
                                (adj_factors['adjustment_factor_flag'] == 3.0)].rename(columns={'adjustment_factor':'rights_adjust'})
        
        merged_df = pd.merge(merged_df,
                            rights_adj[['ticker','adjustment_date','rights_adjust']],
                            left_on=['ticker','ex_date'],
                            right_on=['ticker','adjustment_date'],
                            how='left',
                        )
        merged_df.loc[rights_idx, 'adjustment_factor'] = merged_df.loc[rights_idx, 'rights_adjust']

    merged_df['adjustment_date'] = merged_df['ex_date']
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    merged_df.to_csv(output_filename, index=False)