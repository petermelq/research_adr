import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List
from trade_types import Trade
import pandas_market_calendars as mcal
import pickle as pkl
from strategies.BaseStrategy import BaseStrategy
import numba as nb

@nb.njit(cache=True, fastmath=True)
def recurse_trade_size(trade_size, trade_price, target, ask_px, bid_px, p_volume, pos_0=0):
    n = trade_size.size
    y = np.empty(n, dtype=np.int32)
    prev_pos = pos_0
    for i in range(n):
        desired_trade = target[i] - prev_pos
        trade_sign = np.sign(desired_trade)
        if trade_sign > 0 and trade_price[i] > bid_px[i]:
            final_trade = 0
        elif trade_sign < 0 and trade_price[i] < ask_px[i]:
            final_trade = 0
        else:
            max_trade = p_volume * trade_size[i]
            final_trade = trade_sign * min(max_trade, np.abs(desired_trade))

        prev_pos += final_trade
        y[i] = final_trade
        
    return y

class ADR_passive(BaseStrategy):
    """
    Simulates a ADR passive trading strategy
    """
    def __init__(self,
                adr_info_filename: str,
                lmbda: float,
                p_volume: float,
                ewm_halflife: int,
                start_offset_minutes: int,
                start_offset_hours: int,
                end_offset_minutes: int,
                end_offset_hours: int,
                vol_lookback: int,
                win_multiplier: int = 10,
        ):
        """
        Initialize the minimal strategy.
        
        Args:
            lmbda (float): parameter for strategy
            adr_info_filename (str): path to ADR info CSV file
        """
        super().__init__()
        self.lmbda = lmbda
        self.adr_info = pd.read_csv(adr_info_filename)
        self.ewm_halflife = ewm_halflife
        self.win_size = ewm_halflife * win_multiplier
        self.p_volume = p_volume
        self.start_offset = pd.Timedelta(hours=start_offset_hours, minutes=start_offset_minutes)
        self.end_offset = pd.Timedelta(hours=end_offset_hours, minutes=end_offset_minutes)
        self.vol_lookback = vol_lookback

    def generate_trades(self,
                        current_position: Dict[str, float],
                        trading_day: date,
                        domestic_close_adr,
                        under_close,
                        futures_bars,
                        etf_bbo,
                        adr_tcbbo,
                        adr_close,
                        fitted_params,
                        sector_etfs,
        ) -> List[Trade]:
        if len(fitted_params) == 0: # will be the case on holidays
            return []

        # change this when moving to multi-country backtest
        futures_bars.loc[:, 'exchange'] = 'XLON'
        ny_close = mcal.get_calendar('XNYS').schedule(start_date=trading_day, end_date=trading_day)['market_close'].dt.tz_convert('America/New_York').iloc[0]
        
        trades = []
        
        adr_dict = self.adr_info[['id','adr']].set_index('id')['adr'].str.replace(' US Equity', '').to_dict()
        under_close = under_close.rename(columns=adr_dict)
        
        under_close = under_close.iloc[-self.win_size:]
        domestic_close_adr = domestic_close_adr.iloc[-self.win_size:]
        
        adj_prem_df = np.log((domestic_close_adr.div(under_close)).dropna(axis=0, how='all')).loc['2020':]
        local_close_alpha = (np.exp(adj_prem_df.ewm(halflife=self.ewm_halflife).mean() - adj_prem_df) - 1).iloc[-1]

        exchanges = self.adr_info['exchange'].unique().tolist()
        close_time = {ex: mcal.get_calendar(ex).schedule(start_date=trading_day, end_date=trading_day)['market_close'].dt.tz_convert('America/New_York').iloc[0] for ex in exchanges}
        close_time['XLON'] += pd.Timedelta(hours=0, minutes=5)  # London auction time 5 minutes after close
        
        ticker_close = self.adr_info[['adr','exchange']].copy()
        ticker_close['close_time'] = ticker_close['exchange'].map(close_time)
        ticker_close['adr'] = ticker_close['adr'].str.replace(' US Equity', '')

        merged_futures = futures_bars.merge(ticker_close, on='exchange')
        local_close_fut = merged_futures[merged_futures['timestamp'] == merged_futures['close_time']][['close_time','adr','close']]
        
        etf_bbo = pd.concat(etf_bbo).reset_index(names=['ticker','ts_recv'])
        etf_bbo['mid'] = (etf_bbo['nbbo_bid'] + etf_bbo['nbbo_ask']) / 2
        etf_bbo['exchange'] = 'XLON'
        merged_etf = ticker_close.merge(sector_etfs, on='adr')
        merged_etf = merged_etf.merge(etf_bbo[['ts_recv', 'ticker','exchange','mid']],
                                        left_on=['hedge','exchange'],
                                        right_on=['ticker','exchange'],
                                    )

        local_close_fut = merged_futures[merged_futures['timestamp'] == merged_futures['close_time']][['adr','close']].rename(columns={'close':'local_close_fut'})
        merged_futures = merged_futures[['timestamp','adr','close']].merge(local_close_fut, on='adr')

        local_close_etf = merged_etf[merged_etf['ts_recv'] == merged_etf['close_time']][['adr','ticker','mid']].rename(columns={'mid':'local_close_etf'})
        merged_etf = merged_etf[['ts_recv','adr','ticker','mid','close_time']].merge(local_close_etf, on=['adr','ticker']).rename(columns={'ts_recv':'timestamp'})
        # combine futures and etf data
        merged_data = merged_futures.merge(merged_etf, on=['timestamp','adr'])
        merged_data = merged_data.merge(fitted_params.reset_index(names='adr'), on='adr')

        merged_data['index_ret'] = (merged_data['close'] - merged_data['local_close_fut'])/merged_data['local_close_fut']
        merged_data['sector_ret'] = (merged_data['mid'] - merged_data['local_close_etf'])/merged_data['local_close_etf']

        merged_data['pred_under_ret'] = merged_data['market_beta'] * merged_data['index_ret'] + (merged_data['sector_beta'] * merged_data['sector_ret']).fillna(0)
        merged_data = merged_data.merge(domestic_close_adr.iloc[-1].rename('domestic_close_adr'), left_on='adr', right_index=True)
        merged_data = merged_data.merge(local_close_alpha.rename('local_close_alpha'), left_on='adr', right_index=True)

        for ticker in adr_tcbbo:
            ticker_merged = merged_data[merged_data['adr'] == ticker]
            adr_tcbbo[ticker].index = adr_tcbbo[ticker].index.astype('datetime64[us, America/New_York]')#adr_tcbbo[ticker].index.tz_convert('America/New_York')
            ticker_merged = pd.merge_asof(adr_tcbbo[ticker],
                                            ticker_merged, 
                                            left_index=True,
                                            right_on='timestamp',
                                            direction='backward',
                                        )
            if len(ticker_merged) == 0:
                continue

            ticker_merged = ticker_merged[(ticker_merged.index >= ticker_merged['close_time'] + self.start_offset) & 
                                            (ticker_merged.index < ny_close - self.end_offset)]
            ticker_merged['adr_mid'] = (ticker_merged['ask_px_00'] + ticker_merged['bid_px_00']) / 2
            ticker_merged['ask_ret'] = (ticker_merged['ask_px_00'] - ticker_merged['domestic_close_adr'])/ticker_merged['domestic_close_adr']
            ticker_merged['bid_ret'] = (ticker_merged['bid_px_00'] - ticker_merged['domestic_close_adr'])/ticker_merged['domestic_close_adr']
            
            ticker_merged['ask_alpha'] = ticker_merged['local_close_alpha'] - ticker_merged['ask_ret'] + ticker_merged['pred_under_ret']
            ticker_merged['bid_alpha'] = ticker_merged['local_close_alpha'] - ticker_merged['bid_ret'] + ticker_merged['pred_under_ret']

            ticker_merged['alpha'] = ticker_merged.apply(lambda row: row['bid_alpha'] 
                                                            if np.abs(row['ask_alpha']) < np.abs(row['bid_alpha'])
                                                            else row['ask_alpha'], axis=1)
            sigma = adr_close[ticker].iloc[-self.vol_lookback:].pct_change().std()
            ticker_merged['target'] = (ticker_merged['alpha'] / (self.lmbda * sigma**2)) / ticker_merged['adr_mid']
            ticker_merged['trade_size'] = recurse_trade_size(ticker_merged['size'].to_numpy(dtype=np.float32),
                                                                ticker_merged['price'].to_numpy(dtype=np.float32),
                                                                ticker_merged['target'].to_numpy(dtype=np.float32),
                                                                ticker_merged['ask_px_00'].to_numpy(dtype=np.float32),
                                                                ticker_merged['bid_px_00'].to_numpy(dtype=np.float32),
                                                                self.p_volume,
                                                                pos_0=current_position.get(ticker, 0),
                                                            )

            for ts, row in ticker_merged.iterrows():
                if row['trade_size'] != 0:
                    trades.append(
                        Trade(
                            timestamp=ts,
                            ticker=ticker,
                            size=int(row['trade_size']),
                            price=row['price'],
                        )
                    )
                else:
                    continue

            final_size = ticker_merged['trade_size'].sum()
            final_trade = -(current_position.get(ticker, 0) + final_size)

            if final_trade != 0:
                trades.append(
                    Trade(
                        timestamp=ny_close,
                        ticker=ticker,
                        size=int(final_trade),
                        price=adr_close[ticker].iloc[-1],
                    )
                )
        
        return trades