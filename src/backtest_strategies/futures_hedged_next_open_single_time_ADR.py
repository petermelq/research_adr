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


def _series_at_target_timestamps(series, target_times, tolerance):
    if len(series) == 0 or len(target_times) == 0:
        return pd.Series(dtype=float)

    target_df = target_times.dropna().rename("target_time").sort_values().reset_index()
    if target_df.empty:
        return pd.Series(dtype=float)
    target_df.columns = ["trade_date", "target_time"]
    target_df["target_time"] = _coerce_ny_ns(target_df["target_time"])
    target_df = target_df.sort_values("target_time").reset_index(drop=True)

    price_df = series.sort_index().rename("price").reset_index()
    price_df.columns = ["timestamp", "price"]
    price_df["timestamp"] = _coerce_ny_ns(price_df["timestamp"])
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        target_df,
        price_df,
        left_on="target_time",
        right_on="timestamp",
        direction="backward",
        tolerance=tolerance,
    )
    out = merged.dropna(subset=["price"]).set_index("trade_date")["price"].astype(float).sort_index()
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out[~out.index.duplicated(keep="last")]


def _build_next_open_targets(open_none, open_all, open_times, ratio_atol):
    df = pd.concat({"open_none": open_none, "open_all": open_all}, axis=1).dropna().sort_index()
    if df.empty:
        return pd.DataFrame(columns=["next_open_adj", "exit_multiplier", "next_open_time"])

    df["cum_adj"] = df["open_all"] / df["open_none"]
    next_date = df.index.to_series().shift(-1)
    valid = next_date.eq(df.index.to_series() + pd.Timedelta(days=1))
    out = df.loc[valid, ["open_none", "cum_adj"]].copy()
    out["next_date"] = pd.DatetimeIndex(next_date.loc[out.index])
    out["next_open_none"] = df["open_none"].shift(-1).loc[out.index]
    out["next_cum_adj"] = df["cum_adj"].shift(-1).loc[out.index]

    ratio = out["next_cum_adj"] / out["cum_adj"]
    ratio = ratio.where(~np.isclose(ratio, 1.0, atol=ratio_atol, rtol=0.0), 1.0)
    out["exit_multiplier"] = ratio
    out["next_open_adj"] = out["next_open_none"] * out["exit_multiplier"]

    next_open_time = open_times.reindex(out["next_date"])
    next_open_time = _coerce_ny_ns(next_open_time.to_numpy()).tz_localize(None)
    out["next_open_time"] = pd.Series(next_open_time, index=out.index)
    return out[["next_open_adj", "exit_multiplier", "next_open_time"]]


class futures_hedged_next_open_single_time_ADR(BaseStrategy):
    def __init__(
        self,
        adr_info_filename: str,
        ord_open_none_filename: str,
        ord_open_all_filename: str,
        betas_filename: str,
        futures_symbols_filename: str,
        futures_dir: str,
        var_penalty: float,
        p_volume: float,
        vol_lookback: int,
        turnover_lookback: int,
        skip_earnings: bool = False,
        hedged: bool = True,
        position_limit: float = 1e6,
        next_open_ratio_atol: float = 1e-5,
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
        self.next_open_ratio_atol = next_open_ratio_atol

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

        ord_to_adr = self.adr_info.set_index("id")["adr_ticker"].to_dict()
        self.ord_open_none = (
            pd.read_csv(ord_open_none_filename, index_col=0, parse_dates=True).sort_index().rename(columns=ord_to_adr)
        )
        self.ord_open_all = (
            pd.read_csv(ord_open_all_filename, index_col=0, parse_dates=True).sort_index().rename(columns=ord_to_adr)
        )
        self.betas = pd.read_csv(betas_filename, index_col=0, parse_dates=True).sort_index()

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
        self.exchange_map = self.adr_info.set_index("adr_ticker")["exchange"].to_dict()

        self.open_times_by_exchange = {}
        start_date = min(self.ord_open_none.index.min(), self.betas.index.min()).strftime("%Y-%m-%d")
        end_date = max(self.ord_open_none.index.max(), self.betas.index.max()).strftime("%Y-%m-%d")
        for exchange in self.adr_info["exchange"].dropna().unique():
            sched = mcal.get_calendar(exchange).schedule(start_date=start_date, end_date=end_date)
            self.open_times_by_exchange[exchange] = sched["market_open"].dt.tz_convert("America/New_York")

        unique_future_symbols = sorted(
            sym for sym in pd.Series(self.future_symbol_map).dropna().astype(str).unique().tolist() if sym
        )
        self.future_close_by_symbol = {}
        self.future_entry_by_symbol = {}
        for symbol in unique_future_symbols:
            future_path = os.path.join(futures_dir, f"symbol={symbol}", f"{symbol}_close_to_usd_1min.parquet")
            if not os.path.exists(future_path):
                continue
            close_series = _load_futures_close_series(future_path)
            self.future_close_by_symbol[symbol] = close_series
            self.future_entry_by_symbol[symbol] = _last_price_at_or_before(
                close_series,
                self.trade_time_str,
                self.entry_lookback,
            )

        future_entry = {}
        future_next_open = {}
        next_open_price = {}
        next_open_time = {}
        next_open_multiplier = {}

        for ticker in self.adr_info["adr_ticker"]:
            exchange = self.exchange_map.get(ticker)
            future_symbol = self.future_symbol_map.get(ticker)
            if exchange not in self.open_times_by_exchange:
                continue
            if future_symbol not in self.future_close_by_symbol:
                continue
            if ticker not in self.ord_open_none.columns or ticker not in self.ord_open_all.columns:
                continue

            next_open_targets = _build_next_open_targets(
                self.ord_open_none[ticker].dropna().astype(float),
                self.ord_open_all[ticker].dropna().astype(float),
                self.open_times_by_exchange[exchange],
                ratio_atol=self.next_open_ratio_atol,
            )
            if next_open_targets.empty:
                continue

            future_exit = _series_at_target_timestamps(
                self.future_close_by_symbol[future_symbol],
                next_open_targets["next_open_time"],
                self.entry_lookback,
            )
            if future_exit.empty:
                continue

            next_open_targets = next_open_targets.loc[next_open_targets.index.intersection(future_exit.index)]
            if next_open_targets.empty:
                continue

            next_open_price[ticker] = next_open_targets["next_open_adj"]
            next_open_time[ticker] = next_open_targets["next_open_time"]
            next_open_multiplier[ticker] = next_open_targets["exit_multiplier"]
            future_next_open[ticker] = future_exit

            future_entry_series = self.future_entry_by_symbol.get(future_symbol, pd.Series(dtype=float))
            future_entry[ticker] = future_entry_series.loc[
                future_entry_series.index.intersection(next_open_targets.index)
            ]

        self.next_open_exit = pd.DataFrame(next_open_price).sort_index()
        self.next_open_timestamp = pd.DataFrame(next_open_time).sort_index()
        self.next_open_multiplier = pd.DataFrame(next_open_multiplier).sort_index()
        self.future_next_open_exit = pd.DataFrame(future_next_open).sort_index()
        self.future_entry_price = pd.DataFrame(future_entry).sort_index()

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
            self.next_open_exit.reindex(index=[trade_date], columns=cols).iloc[0].rename("next_open_exit"),
            self.future_entry_price.reindex(index=[trade_date], columns=cols).iloc[0].rename("future_entry_price"),
            self.future_next_open_exit.reindex(index=[trade_date], columns=cols).iloc[0].rename("future_next_open_exit"),
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
        stock_exit_hist = self.next_open_exit.reindex(index=history_idx, columns=cols)
        future_entry_hist = self.future_entry_price.reindex(index=history_idx, columns=cols)
        future_exit_hist = self.future_next_open_exit.reindex(index=history_idx, columns=cols)
        beta_hist = self.betas.reindex(index=history_idx, columns=cols)

        stock_ret = stock_exit_hist / adr_entry_hist - 1.0
        if self.hedged:
            future_ret = future_exit_hist / future_entry_hist - 1.0
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
        shares = (weights["weight"] / weights["trade_price"]).round()

        if shares.isnull().any():
            missing = shares[shares.isnull()].index.tolist()
            raise RuntimeError(f"NaN ADR share counts for {trade_date.date()}: {missing}")

        trades = []
        for ticker in tickers:
            if trade_date.strftime("%Y-%m-%d") == "2025-06-25" and ticker in ["BP", "SHEL"]:
                continue

            exit_timestamp = self.next_open_timestamp.at[trade_date, ticker]
            exit_price = self.next_open_exit.at[trade_date, ticker]
            if pd.isna(exit_timestamp) or pd.isna(exit_price):
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
                    timestamp=pd.Timestamp(exit_timestamp),
                    ticker=ticker,
                    size=-int(shares[ticker]),
                    price=exit_price,
                )
            )

            if not self.hedged:
                continue

            future_symbol = self.future_symbol_map.get(ticker)
            future_entry = self.future_entry_price.at[trade_date, ticker]
            future_exit = self.future_next_open_exit.at[trade_date, ticker]
            beta = self.betas.at[trade_date, ticker]
            if pd.isna(future_entry) or pd.isna(future_exit) or pd.isna(beta):
                continue

            future_contracts = int(np.round((-weights.at[ticker, "weight"] * beta) / future_entry))
            if future_contracts == 0:
                continue

            trades.append(
                Trade(
                    timestamp=trade_date + self.trade_time,
                    ticker=future_symbol,
                    size=future_contracts,
                    price=future_entry,
                )
            )
            trades.append(
                Trade(
                    timestamp=pd.Timestamp(exit_timestamp),
                    ticker=future_symbol,
                    size=-future_contracts,
                    price=future_exit,
                )
            )

        return trades
