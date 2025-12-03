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

class ADR_passive_tick(BaseStrategy):
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
                use_exp_initial: bool = True,
                save_signal: bool = True,
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
        self.use_initial_exp = use_exp_initial
        self.save_signal = save_signal

    def generate_trades(self,
                        current_position: Dict[str, float],
                        trading_day: date,
                        domestic_close_adr_adj,
                        domestic_close_adr_unadj,
                        under_close,
                        futures_ticks,
                        etf_ticks,
                        adr_tcbbo,
                        adr_close,
                        fitted_params,
                        sector_etfs,
        ) -> List[Trade]:
        
        if len(fitted_params) == 0: # will be the case on holidays
            return []

        # change this when moving to multi-country backtest
        futures_ticks.loc[:, 'exchange'] = 'XLON'
        ny_open = (
                    mcal.get_calendar('XNYS')
                    .schedule(start_date=trading_day, end_date=trading_day)['market_open']
                    .dt.tz_convert('America/New_York').iloc[0]
                )
        ny_close = (
                    mcal.get_calendar('XNYS')
                    .schedule(start_date=trading_day, end_date=trading_day)['market_close']
                    .dt.tz_convert('America/New_York').iloc[0]
        )
        trades = []
        
        adr_dict = self.adr_info[['id','adr']].set_index('id')['adr'].str.replace(' US Equity', '').to_dict()
        under_close = under_close.rename(columns=adr_dict)
        
        under_close = under_close.iloc[-self.win_size:]
        domestic_close_adr_adj = domestic_close_adr_adj.iloc[-self.win_size:]
        
        adj_prem_df = np.log((domestic_close_adr_adj.div(under_close)).dropna(axis=0, how='all')).loc['2020':]
        local_close_alpha = (np.exp(adj_prem_df.ewm(halflife=self.ewm_halflife).mean() - adj_prem_df) - 1).iloc[-1]

        exchanges = self.adr_info['exchange'].unique().tolist()
        close_time = {ex: mcal.get_calendar(ex).schedule(start_date=trading_day, end_date=trading_day)['market_close'].dt.tz_convert('America/New_York').iloc[0] for ex in exchanges}
        close_time['XLON'] += pd.Timedelta(hours=0, minutes=5)  # London auction time 5 minutes after close
        
        ticker_close = self.adr_info[['adr','exchange']].copy()
        ticker_close['close_time'] = ticker_close['exchange'].map(close_time)
        ticker_close['adr'] = ticker_close['adr'].str.replace(' US Equity', '')

        futures_ticks.index = futures_ticks.index.tz_convert('America/New_York')
        futures_ticks = futures_ticks.reset_index(names='timestamp')
        futures_ticks = futures_ticks[(futures_ticks['timestamp'] >= ny_open) & (futures_ticks['timestamp'] <= ny_close)]
        merged_futures = futures_ticks.merge(ticker_close, on='exchange')
        
        etf_ticks = pd.concat(etf_ticks).reset_index(names=['ticker','ts_recv'])
        etf_ticks = etf_ticks[(etf_ticks['ts_recv'] >= ny_open) & (etf_ticks['ts_recv'] <= ny_close)]
        etf_ticks['ts_recv'] = etf_ticks['ts_recv'].dt.tz_convert('America/New_York')

        etf_ticks['mid'] = (etf_ticks['bid_px'] + etf_ticks['ask_px']) / 2
        etf_ticks['exchange'] = 'XLON'
        merged_etf = ticker_close.merge(sector_etfs, on='adr')
        merged_etf = merged_etf.merge(etf_ticks[['ts_recv', 'ticker','exchange','mid']],
                                        left_on=['hedge','exchange'],
                                        right_on=['ticker','exchange'],
                                    )

        local_close_fut = (
                            merged_futures.groupby('adr')
                            .apply(lambda _df:_df[_df['timestamp'] < _df['close_time']].iloc[[-1]])
                            .reset_index(drop=True)[['adr','mid_futures']]
                            .rename(columns={'mid_futures':'local_close_fut'})
        )
        merged_futures = merged_futures[['timestamp','adr','mid_futures','close_time']].merge(local_close_fut, on='adr')
        
        local_close_etf = (
                            merged_etf.groupby('adr')
                            .apply(lambda _df:_df[_df['ts_recv'] < _df['close_time']].iloc[-1])
                            .reset_index(drop=True)
                            .rename(columns={'mid':'local_close_etf'})[['adr','ticker','local_close_etf']]
        )
        merged_etf = merged_etf[['ts_recv','adr','ticker','mid']].merge(local_close_etf, on=['adr','ticker']).rename(columns={'ts_recv':'timestamp','mid':'mid_etf'})
        if self.save_signal:
            intraday_exp_return_dirname = '/home/pmalonis/adr_trade/data/processed/matching_hedged_intraday_exp_return'
            if not os.path.exists(intraday_exp_return_dirname):
                os.makedirs(intraday_exp_return_dirname, exist_ok=True)

        date_str = trading_day.strftime('%Y-%m-%d')

        # combine futures and etf data
        for ticker in adr_tcbbo:
            if trading_day.strftime('%Y-%m-%d') == '2025-06-25' and ticker in ['BP','SHEL']:
                continue

            ticker_futures = merged_futures[merged_futures['adr']==ticker].groupby('timestamp').last().drop(columns=['adr'])
            ticker_etf = merged_etf[merged_etf['adr']==ticker].groupby('timestamp').last().drop(columns=['adr','ticker'])
            ticker_merged = pd.concat([ticker_futures,
                                        ticker_etf],
                                        axis=1)
            
            #ticker_merged.index = ticker_merged.index.tz_convert('America/New_York')
            ticker_merged = ticker_merged.ffill()

            ticker_merged['index_ret'] = (ticker_merged['mid_futures'] - ticker_merged['local_close_fut'])/ticker_merged['local_close_fut']
            ticker_merged['sector_ret'] = (ticker_merged['mid_etf'] - ticker_merged['local_close_etf'])/ticker_merged['local_close_etf']
            ticker_merged['pred_under_ret'] = fitted_params.loc[ticker,'market_beta'] * ticker_merged['index_ret'] + (fitted_params.loc[ticker,'sector_beta'] * ticker_merged['sector_ret']).fillna(0)
            ticker_merged['domestic_close_adr'] = domestic_close_adr_unadj[ticker].iloc[-1]
            
            adr_tcbbo[ticker].index = adr_tcbbo[ticker].index.astype('datetime64[ns, America/New_York]')#adr_tcbbo[ticker].index.tz_convert('America/New_York')

            ticker_merged = pd.merge_asof(adr_tcbbo[ticker],
                                            ticker_merged,
                                            left_index=True,
                                            right_index=True,
                                            direction='backward',
                                        )
            if len(ticker_merged) == 0:
                continue
            
            ticker_merged['adr_mid'] = (ticker_merged['ask_px_00'] + ticker_merged['bid_px_00']) / 2
            domestic_close_ask = ticker_merged[ticker_merged.index < ticker_merged['close_time']].iloc[-1]['ask_px_00']
            domestic_close_bid = ticker_merged[ticker_merged.index < ticker_merged['close_time']].iloc[-1]['bid_px_00']
            ticker_merged['ask_ret'] = (ticker_merged['ask_px_00'] - domestic_close_ask)/domestic_close_ask
            ticker_merged['bid_ret'] = (ticker_merged['bid_px_00'] - domestic_close_bid)/domestic_close_bid
            
            # ticker_merged['ask_ret'] = (ticker_merged['ask_px_00'] - ticker_merged['domestic_close_adr'])/ticker_merged['domestic_close_adr']
            # ticker_merged['bid_ret'] = (ticker_merged['bid_px_00'] - ticker_merged['domestic_close_adr'])/ticker_merged['domestic_close_adr']
            
            # limit to ny hours post domestic close
            ticker_merged = ticker_merged[(ticker_merged.index >= ticker_merged['close_time']) &
                                            (ticker_merged.index < ny_close)]

            # computing alphas
            ticker_merged['local_close_alpha'] = local_close_alpha[ticker]
            if self.use_initial_exp:
                ticker_merged['ask_alpha'] = ticker_merged['local_close_alpha'] - ticker_merged['ask_ret'] + ticker_merged['pred_under_ret']
                ticker_merged['bid_alpha'] = ticker_merged['local_close_alpha'] - ticker_merged['bid_ret'] + ticker_merged['pred_under_ret']
            else:
                ticker_merged['ask_alpha'] = - ticker_merged['ask_ret'] + ticker_merged['pred_under_ret']
                ticker_merged['bid_alpha'] = - ticker_merged['bid_ret'] + ticker_merged['pred_under_ret']

            ticker_merged['alpha'] = ticker_merged.apply(lambda row: row['bid_alpha']
                                                        if np.abs(row['ask_alpha']) < np.abs(row['bid_alpha'])
                                                        else row['ask_alpha'], axis=1)
            exp_return_filename = os.path.join(intraday_exp_return_dirname,
                                                f'date={date_str}',
                                                f'ticker={ticker}',
                                                'data.parquet')
            
            if self.save_signal:
                os.makedirs(os.path.dirname(exp_return_filename), exist_ok=True)
                ticker_merged[['ask_ret','bid_ret','pred_under_ret','local_close_alpha']].rename(columns={'pred_under_ret': 'pred_ord_ret'}).to_parquet(exp_return_filename, compression='snappy')

            if ticker in ['BP','SHEL'] and trading_day > pd.Timestamp('2025-06-25'):
                sigma = adr_close[ticker].iloc[-self.vol_lookback-1:-1].drop(index=pd.Timestamp('2025-06-25')).pct_change().std()
            else:
                sigma = adr_close[ticker].iloc[-self.vol_lookback-1:-1].pct_change().std()

            # limiting to trading window
            ticker_merged = ticker_merged[(ticker_merged.index >= ticker_merged['close_time'] + self.start_offset) &
                                            (ticker_merged.index < ny_close - self.end_offset)]

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