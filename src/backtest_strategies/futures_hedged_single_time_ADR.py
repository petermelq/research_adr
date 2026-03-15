import os
import sys
from datetime import date
from typing import Dict, List

import cvxpy as cp
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from backtester.strategies import BaseStrategy
from backtester.trade_types import Trade

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(MODULE_DIR, ".."))
import utils


def get_normal_close_time(exchange):
    return mcal.get_calendar(exchange).close_time


def _coerce_ny_ns(index_like):
    idx = pd.DatetimeIndex(index_like)
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    else:
        idx = idx.tz_convert("America/New_York")
    return idx.as_unit("ns")


def _load_futures_close_series(path):
    fut_df = pd.read_parquet(path, columns=["timestamp", "close"]).dropna(subset=["close"])
    fut_df["timestamp"] = pd.to_datetime(fut_df["timestamp"])
    fut_df = fut_df.sort_values("timestamp").set_index("timestamp")
    fut_df.index = _coerce_ny_ns(fut_df.index)
    return fut_df["close"].astype(float)


def _last_price_at_or_before(series, target_time_str, lookback):
    start_time = (pd.Timestamp(target_time_str) - lookback).strftime("%H:%M")
    window = series.between_time(start_time, target_time_str)
    if len(window) == 0:
        return pd.Series(dtype=float)
    by_date = pd.Series(
        window.to_numpy(dtype=float),
        index=window.index.tz_convert("America/New_York").tz_localize(None).normalize(),
    )
    return by_date.groupby(level=0).last().sort_index()


class futures_hedged_single_time_ADR(BaseStrategy):
    def __init__(
        self,
        adr_info_filename: str,
        betas_filename: str,
        futures_symbols_filename: str,
        futures_dir: str,
        futures_ny_close_filename: str,
        var_penalty: float,
        p_volume: float,
        vol_lookback: int,
        turnover_lookback: int,
        skip_earnings: bool = False,
        hedged: bool = True,
        position_limit: float = 1e6,
        trade_time_hours: int = None,
        trade_time_min: int = None,
    ):
        super().__init__()
        self.var_penalty = var_penalty
        self.p_volume = p_volume
        self.vol_lookback = vol_lookback
        self.turnover_lookback = turnover_lookback
        self.hedged = hedged
        self.position_limit = position_limit
        self.skip_earnings = skip_earnings
        self.entry_lookback = pd.Timedelta("30min")

        params = utils.load_params()
        if trade_time_hours is None:
            trade_time_hours = params["fixed_trade_time_hours"]
        if trade_time_min is None:
            trade_time_min = params["fixed_trade_time_min"]
        self.trade_time = pd.Timedelta(
            hours=trade_time_hours,
            minutes=trade_time_min,
        )
        self.trade_time_str = f"{trade_time_hours:02d}:{trade_time_min:02d}"
        self.close_time = pd.Timedelta("16:00:00")

        self.adr_info = pd.read_csv(adr_info_filename)
        self.adr_info["adr_ticker"] = self.adr_info["adr"].str.replace(" US Equity", "", regex=False)
        self.adr_info["index_future_bbg"] = self.adr_info["index_future_bbg"].astype(str).str.strip()
        self.adr_info["normal_close_time"] = self.adr_info["exchange"].apply(get_normal_close_time)

        schedules = []
        exchanges = self.adr_info["exchange"].dropna().unique().tolist() + ["XNYS"]
        for exchange in exchanges:
            cal = mcal.get_calendar(exchange)
            sched = cal.schedule("1980-01-01", "2030-01-01").rename(columns={"market_close": exchange})
            schedules.append(sched[[exchange]])

        self.schedule = pd.concat(schedules, axis=1)
        for exchange in exchanges:
            self.schedule.loc[self.schedule[exchange].isnull(), exchange] = (
                self.schedule.loc[self.schedule[exchange].isnull()].index.tz_localize("UTC")
            )
            cal = mcal.get_calendar(exchange)
            self.schedule[exchange] = self.schedule[exchange].dt.tz_convert(cal.tz).dt.time

        if skip_earnings:
            self.earnings = pd.read_csv(
                os.path.join(MODULE_DIR, "..", "..", "data", "raw", "earnings.csv"),
                index_col=0,
                parse_dates=["announcement_date"],
            )
            adr_dict = self.adr_info.set_index("id")["adr"].to_dict()
            self.earnings.index = [adr_dict[s].split()[0] for s in self.earnings.index]

        self.betas = pd.read_csv(betas_filename, index_col=0, parse_dates=True).sort_index()
        self.future_ny_close = pd.read_csv(futures_ny_close_filename, index_col=0, parse_dates=True).sort_index()

        futures_symbols = pd.read_csv(futures_symbols_filename)
        futures_symbols["bloomberg_symbol"] = futures_symbols["bloomberg_symbol"].astype(str).str.strip()
        futures_symbols["first_rate_symbol"] = futures_symbols["first_rate_symbol"].astype(str).str.strip()
        merged = self.adr_info.merge(
            futures_symbols[["bloomberg_symbol", "first_rate_symbol"]],
            left_on="index_future_bbg",
            right_on="bloomberg_symbol",
            how="left",
        )
        self.future_symbol_map = merged.set_index("adr_ticker")["first_rate_symbol"].to_dict()

        unique_future_symbols = sorted(
            sym for sym in pd.Series(self.future_symbol_map).dropna().astype(str).unique().tolist() if sym
        )
        self.future_entry_by_symbol = {}
        for symbol in unique_future_symbols:
            future_path = os.path.join(futures_dir, f"symbol={symbol}", f"{symbol}_close_to_usd_1min.parquet")
            if not os.path.exists(future_path):
                continue
            close_series = _load_futures_close_series(future_path)
            self.future_entry_by_symbol[symbol] = _last_price_at_or_before(
                close_series,
                self.trade_time_str,
                self.entry_lookback,
            )

        future_entry = {}
        future_close = {}
        for ticker in self.adr_info["adr_ticker"]:
            future_symbol = self.future_symbol_map.get(ticker)
            if future_symbol not in self.future_entry_by_symbol:
                continue
            if future_symbol not in self.future_ny_close.columns:
                continue
            future_entry[ticker] = self.future_entry_by_symbol[future_symbol]
            future_close[ticker] = self.future_ny_close[future_symbol]

        self.future_entry_price = pd.DataFrame(future_entry).sort_index()
        self.future_close_price = pd.DataFrame(future_close).sort_index()

    def normal_close_tickers(self, trading_day):
        merged = pd.merge(
            self.adr_info[["adr", "normal_close_time", "exchange"]],
            self.schedule.loc[trading_day].rename("day_close_time"),
            left_on="exchange",
            right_index=True,
        )
        return merged[merged["day_close_time"] == merged["normal_close_time"]]["adr"].str.replace(" US Equity", "").tolist()

    def ny_normal_close(self, trading_day):
        return self.schedule.loc[trading_day, "XNYS"] == mcal.get_calendar("XNYS").close_time

    def generate_trades(
        self,
        current_position: Dict[str, float],
        trading_day: date,
        adr_trade_price: pd.DataFrame,
        adr_signal: pd.DataFrame,
        adr_close: pd.DataFrame,
        turnover_df: pd.DataFrame,
    ) -> List[Trade]:
        del current_position
        trade_date = pd.Timestamp(trading_day)
        trading_tickers = self.normal_close_tickers(trading_day)

        if (
            not self.ny_normal_close(trading_day)
            or trade_date not in adr_signal.index
            or len(trading_tickers) < adr_trade_price.shape[1]
        ):
            return []

        cols = adr_trade_price.iloc[-1].dropna().index.intersection(adr_signal.columns)
        cols = pd.Index([c for c in cols if c in trading_tickers])
        if self.skip_earnings:
            cols = pd.Index(
                [c for c in cols if c not in self.earnings[self.earnings["announcement_date"] == trade_date].index]
            )

        required_today = [
            adr_trade_price.loc[trade_date, cols].rename("adr_trade_price"),
            adr_close.reindex(index=[trade_date], columns=cols).iloc[0].rename("adr_close"),
            self.future_entry_price.reindex(index=[trade_date], columns=cols).iloc[0].rename("future_entry_price"),
            self.future_close_price.reindex(index=[trade_date], columns=cols).iloc[0].rename("future_close_price"),
            self.betas.reindex(index=[trade_date], columns=cols).iloc[0].rename("beta"),
        ]
        valid_today = pd.concat(required_today, axis=1).notna().all(axis=1)
        cols = cols[valid_today.values]
        if len(cols) == 0:
            return []

        adr_signal = adr_signal[cols].dropna(how="all", axis=1)
        if adr_signal.empty or trade_date not in adr_signal.index:
            return []

        history_signal = adr_signal.iloc[-self.vol_lookback - 1 : -1]
        history_idx = history_signal.index
        if len(history_idx) == 0:
            return []

        adr_entry_hist = adr_trade_price.reindex(index=history_idx, columns=cols)
        stock_exit_hist = adr_close.reindex(index=history_idx, columns=cols)
        future_entry_hist = self.future_entry_price.reindex(index=history_idx, columns=cols)
        future_exit_hist = self.future_close_price.reindex(index=history_idx, columns=cols)
        beta_hist = self.betas.reindex(index=history_idx, columns=cols)

        stock_ret = (stock_exit_hist - adr_entry_hist) / stock_exit_hist
        if self.hedged:
            future_ret = (future_exit_hist - future_entry_hist) / future_exit_hist
            realized_ret = stock_ret - beta_hist * future_ret
        else:
            realized_ret = stock_ret

        res = realized_ret - history_signal

        if pd.Timestamp("2025-06-25") in res.index:
            for ticker in ["BP", "SHEL"]:
                if ticker in res.columns:
                    res.loc[pd.Timestamp("2025-06-25"), ticker] = 0.0

        res = pd.concat([res.loc[: "2025-04-03"], res.loc["2025-04-09" :]])
        res = res[adr_signal.columns]
        covariance = cp.psd_wrap(res.fillna(0).cov().values)

        tickers = adr_signal.columns.tolist()
        alpha = adr_signal.loc[trade_date].clip(lower=-0.01, upper=0.01).fillna(0).values
        turnover = turnover_df.loc[:trade_date, tickers].iloc[-self.turnover_lookback : -1].mean().fillna(1).values

        n_assets = len(tickers)
        w = cp.Variable(n_assets)
        objective = cp.Maximize((alpha @ w) - self.var_penalty * cp.quad_form(w, covariance))
        adv_constraint = cp.multiply(cp.abs(w), 1 / turnover) <= self.p_volume
        pos_constraint = cp.abs(w) <= self.position_limit
        prob = cp.Problem(objective, [adv_constraint, pos_constraint])

        try:
            prob.solve(solver="CLARABEL", max_iter=100000)
        except Exception as exc:
            raise RuntimeError(f"Optimization failed for {trade_date.date()}") from exc

        weights = pd.DataFrame({"weight": w.value}, index=tickers)
        weights["weight"] = weights["weight"].clip(lower=-2e6, upper=2e6)
        trade_price = adr_trade_price.loc[trade_date, tickers]
        weights = weights.merge(trade_price.rename("trade_price"), left_index=True, right_index=True)
        weights = weights.merge(self.betas.loc[trade_date, tickers].rename("beta"), left_index=True, right_index=True)
        weights["future_symbol"] = weights.index.map(self.future_symbol_map)
        weights["future_weight"] = -weights["weight"] * weights["beta"]
        shares = (weights["weight"] / weights["trade_price"]).round()

        if shares.isnull().any():
            missing = shares[shares.isnull()].index.tolist()
            raise RuntimeError(f"NaN ADR share counts for {trade_date.date()}: {missing}")

        trades = []
        for ticker in tickers:
            if trade_date.strftime("%Y-%m-%d") == "2025-06-25" and ticker in ["BP", "SHEL"]:
                continue

            trades.append(
                Trade(
                    timestamp=trade_date + self.trade_time,
                    ticker=ticker,
                    size=int(shares[ticker]),
                    price=trade_price[ticker],
                )
            )
            trades.append(
                Trade(
                    timestamp=trade_date + self.close_time,
                    ticker=ticker,
                    size=-int(shares[ticker]),
                    price=adr_close.loc[trade_date, ticker],
                )
            )

        if not self.hedged:
            return trades

        future_weights = weights.dropna(subset=["future_symbol"]).groupby("future_symbol")["future_weight"].sum()
        future_entry = self.future_ny_close.loc[trade_date, future_weights.index].rename("future_close")
        future_trade_price = pd.Series(
            {
                symbol: self.future_entry_price.at[trade_date, ticker]
                for ticker, symbol in weights["future_symbol"].dropna().items()
                if trade_date in self.future_entry_price.index and ticker in self.future_entry_price.columns
            }
        )
        future_trade_price = future_trade_price[~future_trade_price.index.duplicated(keep="first")]
        future_contracts = (future_weights / future_trade_price.reindex(future_weights.index)).round()

        for symbol, size in future_contracts.items():
            if pd.isna(size) or int(size) == 0:
                continue
            trades.append(
                Trade(
                    timestamp=trade_date + self.trade_time,
                    ticker=symbol,
                    size=int(size),
                    price=future_trade_price[symbol],
                )
            )
            trades.append(
                Trade(
                    timestamp=trade_date + self.close_time,
                    ticker=symbol,
                    size=-int(size),
                    price=future_entry[symbol],
                )
            )

        return trades
