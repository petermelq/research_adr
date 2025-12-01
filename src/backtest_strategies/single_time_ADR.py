import pandas as pd
import numpy as np
import cvxpy as cp
import numba as nb
from datetime import datetime, date
from typing import Dict, List
from backtester.trade_types import Trade
from backtester.strategies import BaseStrategy
from .. import utils

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
        self.adr_info = pd.read_csv(adr_info_filename)
        self.p_volume = p_volume
        self.vol_lookback = vol_lookback
        params = utils.load_params()
        self.trade_time = pd.Timedelta(hours=params['fixed_trade_time_hours'],
                                        minutes=params['fixed_trade_time_min'])

    def generate_trades(self,
                        current_position: Dict[str, float],
                        trading_day: date,
                        adr_trade_price: pd.DataFrame,
                        adr_signal: pd.DataFrame,
                        adr_close: pd.DataFrame,
                        turnover_df: pd.DataFrame,
        ) -> List[Trade]:
        if trading_day not in adr_signal.index:
            return []
        else:
            merged_prices = pd.merge(adr_trade_price.iloc[-self.vol_lookback:].stack().rename('trade_price'),
                                    adr_close.iloc[-self.vol_lookback:].stack().rename('close'),
                                    right_index=True,
                                    left_index=True)
            ret = ((merged_prices['close'] - merged_prices['trade_price'])/merged_prices['close']).unstack()
            res = ret - adr_signal.iloc[-self.vol_lookback:]
            
            if pd.Timestamp('2025-06-25') in res.index:
                res.loc[pd.Timestamp('2025-06-25'), ['BP','SHEL']] = 0.0
            
            Cov = res.cov().values

            tickers = adr_signal.columns.tolist()
            Cov = cp.psd_wrap(Cov)
            alpha = adr_signal.loc[trading_day].fillna(0).values
            turnover = turnover_df.loc[trading_day, tickers].values

            N = Cov.shape[0]
            w = cp.Variable(N)

            objective = cp.Maximize((alpha @ w) - self.var_penalty * cp.quad_form(w, Cov))
            adv_constraint = w @ (1/turnover) <= self.p_volume
            constraints = [adv_constraint]
            
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver='CLARABEL', max_iter=100000)
            weights = pd.DataFrame({'weight': w.value}, index=tickers)
            
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