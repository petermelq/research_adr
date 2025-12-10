import pandas as pd
import numpy as np
import cvxpy as cp
import numba as nb
from datetime import datetime, date
from typing import Dict, List
from backtester.trade_types import Trade
from backtester.strategies import BaseStrategy
import sys
import os
import pandas_market_calendars as mcal
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(MODULE_DIR, '..'))
import utils

def get_day_close_time(exchange, trading_day):
    cal = mcal.get_calendar(exchange)
    sched = cal.schedule(pd.Timestamp(trading_day), pd.Timestamp(trading_day))['market_close']
    if len(sched) == 0:
        close = trading_day.tz_localize('UTC').normalize()
    else:
        close = sched.iloc[0]
    
    return close

def get_normal_close_time(exchange):
    cal = mcal.get_calendar(exchange)
    normal_close = cal.close_time
                
    return normal_close

def get_tz(exchange):
    cal = mcal.get_calendar(exchange)
    tz = cal.tz

    return tz

# def ny_normal_close(trading_day):
#     close = get_day_close_time('NYSE', trading_day)
#     normal_close = get_normal_close_time('NYSE')
#     if normal_close == close.tz_convert('America/New_York').time():
#         return True
#     else:
#         return False

# def normal_close_tickers(adr_info, trading_day):
#     exchange_info = adr_info[['exchange']].drop_duplicates(subset='exchange')
#     exchange_info['day_close_time'] = exchange_info['exchange'].apply(lambda x: get_day_close_time(x, trading_day))
#     exchange_info['normal_close_time'] = exchange_info['exchange'].apply(get_normal_close_time)
#     exchange_info['tz'] = exchange_info['exchange'].apply(get_tz)
#     exchange_info['day_local_close_time'] = exchange_info.apply(lambda x: x['day_close_time'].tz_convert(x['tz']).time(), axis=1)
    
#     merged = pd.merge(adr_info, exchange_info, on='exchange')
#     tickers = merged[merged['day_local_close_time'] == merged['normal_close_time']]['adr'].str.replace(' US Equity', '').tolist()
    
#     return tickers

class single_time_ADR(BaseStrategy):
    """
    Simulates a ADR passive trading strategy
    """
    def __init__(self,
                adr_info_filename: str,
                var_penalty: float,
                p_volume: float,
                vol_lookback: int,
        ):
        """
        Initialize the minimal strategy.
        
        Args:
            lmbda (float): parameter for strategy
            adr_info_filename (str): path to ADR info CSV file
        """
        super().__init__()
        self.var_penalty = var_penalty
        self.p_volume = p_volume
        self.vol_lookback = vol_lookback
        params = utils.load_params()
        self.trade_time = pd.Timedelta(hours=params['fixed_trade_time_hours'],
                                        minutes=params['fixed_trade_time_min'])
        # setting up adr_info
        self.adr_info = pd.read_csv(adr_info_filename)
        schedules = []
        exchanges = self.adr_info['exchange'].unique().tolist() + ['XNYS']
        for exchange in exchanges:
            cal = mcal.get_calendar(exchange)
            sched = cal.schedule('1980-01-01', '2030-01-01').rename(columns={'market_close':exchange})
            schedules.append(sched[[exchange]])

        self.schedule = pd.concat(schedules, axis=1)
        for exchange in exchanges:
            # setting holiday close times (null values) equal to 
            self.schedule.loc[self.schedule[exchange].isnull(), exchange] = self.schedule.loc[self.schedule[exchange].isnull()].index.tz_localize('UTC')
            cal = mcal.get_calendar(exchange)
            self.schedule[exchange] = self.schedule[exchange].dt.tz_convert(cal.tz).dt.time

        self.adr_info['normal_close_time'] = self.adr_info['exchange'].apply(get_normal_close_time)
        self.adr_info['tz'] = self.adr_info['exchange'].apply(get_tz)

    def normal_close_tickers(self, trading_day):
        merged = pd.merge(self.adr_info[['adr','normal_close_time','exchange']],self.schedule.loc['2025-01-02'].rename('day_close_time'),left_on='exchange',right_index=True)
        tickers = merged[merged['day_close_time'] == merged['normal_close_time']]['adr'].str.replace(' US Equity', '').tolist()
        
        return tickers
    
    def ny_normal_close(self, trading_day):
        return self.schedule.loc[trading_day, 'XNYS'] == mcal.get_calendar('XNYS').close_time

    def generate_trades(self,
                        current_position: Dict[str, float],
                        trading_day: date,
                        adr_trade_price: pd.DataFrame,
                        adr_signal: pd.DataFrame,
                        adr_close: pd.DataFrame,
                        turnover_df: pd.DataFrame,
        ) -> List[Trade]:
        
        trading_tickers = self.normal_close_tickers(trading_day)
        
        if (not self.ny_normal_close(trading_day) or
            trading_day not in adr_signal.index or
            len(trading_tickers) < adr_trade_price.shape[1]
        ):
            return []
        else:
            merged_prices = pd.merge(adr_trade_price.iloc[-self.vol_lookback-1:-1].stack().rename('trade_price'),
                                    adr_close.iloc[-self.vol_lookback-1:-1].stack().rename('close'),
                                    right_index=True,
                                    left_index=True)
            
            ret = ((merged_prices['close'] - merged_prices['trade_price'])/merged_prices['close']).unstack()
            res = ret - adr_signal.iloc[-self.vol_lookback-1:-1]

            if pd.Timestamp('2025-06-25') in res.index:
                res.loc[pd.Timestamp('2025-06-25'), ['BP','SHEL']] = 0.0
            
            res = pd.concat([res.loc[:'2025-04-01'],res.loc['2025-04-30':]])
            res = res[adr_signal.columns] # making sure columns are aligned
            Cov = res.dropna().cov().values
            
            # Cov = res.cov().values

            tickers = adr_signal.columns.tolist()
            Cov = cp.psd_wrap(Cov)
            alpha = adr_signal.loc[trading_day].fillna(0).values
            turnover = turnover_df.loc[trading_day, tickers].values

            N = Cov.shape[0]
            w = cp.Variable(N)

            objective = cp.Maximize((alpha @ w) - self.var_penalty * cp.quad_form(w, Cov))
            adv_constraint = cp.multiply(cp.abs(w), 1/turnover) <= self.p_volume
            constraints = [adv_constraint]
            
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver='CLARABEL', max_iter=100000)
            
            weights = pd.DataFrame({'weight': w.value}, index=tickers)
            weights['weight'] = weights['weight'].clip(lower=-2e6, upper=2e6)
            
            trade_price = adr_trade_price.loc[trading_day]
            weights = weights.merge(trade_price.rename('trade_price'), left_index=True, right_index=True)
            
            shares = (weights['weight']/weights['trade_price']).round()
            
            trades = []
            for ticker in tickers:
                if trading_day.strftime('%Y-%m-%d') == '2025-06-25' and ticker in ['BP', 'SHEL']:
                    continue
                else:
                    trades.append(
                                    Trade(
                                        timestamp=trading_day + self.trade_time,
                                        ticker=ticker,
                                        size=int(shares[ticker]),
                                        price=trade_price[ticker],
                                    )
                                )
                    trades.append(
                                    Trade(
                                        timestamp=trading_day + pd.Timedelta('16:00:00'),
                                        ticker=ticker,
                                        size=-int(shares[ticker]),
                                        price=adr_close.loc[trading_day, ticker],
                                    )
                                )
                    
            return trades