from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    return normalized.sort_index()


def _select_trade_columns(
    adr_signal: pd.DataFrame,
    adr_trade_price: pd.DataFrame,
    trade_symbols: Sequence[str] | None,
) -> list[str]:
    cols = adr_trade_price.iloc[-1].dropna().index.intersection(adr_signal.columns)
    if trade_symbols is not None:
        cols = cols.intersection(pd.Index(list(trade_symbols)))
    return cols.tolist()


def compute_backtest_covariance(
    trade_date: str | pd.Timestamp,
    adr_signal: pd.DataFrame,
    adr_trade_price: pd.DataFrame,
    adr_close: pd.DataFrame,
    etf_trade_price: pd.DataFrame,
    etf_close: pd.DataFrame,
    hedge_ratios: pd.DataFrame,
    hedge_map: dict[str, str],
    vol_lookback: int = 100,
    trade_symbols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_date_ts = pd.Timestamp(trade_date)

    adr_signal = _normalize_history_frame(adr_signal).loc[:trade_date_ts]
    adr_trade_price = _normalize_history_frame(adr_trade_price).loc[:trade_date_ts]
    adr_close = _normalize_history_frame(adr_close).loc[:trade_date_ts]
    etf_trade_price = _normalize_history_frame(etf_trade_price).loc[:trade_date_ts]
    etf_close = _normalize_history_frame(etf_close).loc[:trade_date_ts]
    hedge_ratios = _normalize_history_frame(hedge_ratios).loc[:trade_date_ts]

    if trade_date_ts not in adr_signal.index:
        raise KeyError(f"{trade_date_ts.date()} not present in ADR signal history")
    if trade_date_ts not in adr_trade_price.index:
        raise KeyError(f"{trade_date_ts.date()} not present in ADR trade-price history")

    cols = _select_trade_columns(adr_signal, adr_trade_price, trade_symbols)
    if not cols:
        raise ValueError("No overlapping ADR columns available for covariance estimation")

    adr_signal = adr_signal[cols].dropna(how="all", axis=1)
    if adr_signal.empty:
        raise ValueError("Signal matrix is empty after dropping all-null ADR columns")

    history_slice = slice(-vol_lookback - 1, -1)
    merged_prices = pd.merge(
        adr_trade_price.iloc[history_slice].stack().rename("trade_price"),
        adr_close.iloc[history_slice].stack().rename("close"),
        right_index=True,
        left_index=True,
    )
    merged_etf_prices = pd.merge(
        etf_trade_price.iloc[history_slice].stack().rename("trade_price"),
        etf_close.iloc[history_slice].stack().rename("close"),
        right_index=True,
        left_index=True,
    )
    adr_ret = ((merged_prices["close"] - merged_prices["trade_price"]) / merged_prices["close"]).rename("adr_ret")
    etf_ret = ((merged_etf_prices["close"] - merged_etf_prices["trade_price"]) / merged_etf_prices["close"])
    hr_stacked = hedge_ratios.iloc[history_slice].stack().rename("hedge_ratio")
    merged = pd.merge(hr_stacked, adr_ret, left_index=True, right_index=True).reset_index(names=["date", "ticker"])
    merged["hedge_ticker"] = merged["ticker"].map(hedge_map)
    etf_ret = etf_ret.to_frame(name="etf_ret").reset_index(names=["date", "hedge_ticker"])
    merged = merged.merge(etf_ret, on=["date", "hedge_ticker"])
    merged["hedged_ret"] = merged["adr_ret"] - merged["hedge_ratio"] * merged["etf_ret"]
    ret = merged.pivot(index="date", columns="ticker", values="hedged_ret")
    res = ret - adr_signal.iloc[history_slice]

    if pd.Timestamp("2025-06-25") in res.index:
        for ticker in ["BP", "SHEL"]:
            if ticker in res.columns:
                res.loc[pd.Timestamp("2025-06-25"), ticker] = 0.0

    res = pd.concat([res.loc[: "2025-04-03"], res.loc["2025-04-09" :]])
    res = res[adr_signal.columns]
    covariance = res.fillna(0).cov()
    covariance = covariance.reindex(index=adr_signal.columns, columns=adr_signal.columns)
    return covariance, res


__all__ = ["compute_backtest_covariance"]
