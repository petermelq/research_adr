from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from adr_strategy_kernel.common import (
    is_usd_currency,
    normalize_currency,
    resolve_repo_path,
)


def load_index_reference_mappings(
    futures_symbols_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    path = Path(futures_symbols_path) if futures_symbols_path is not None else resolve_repo_path(
        "data", "raw", "futures_symbols.csv", repo_root=repo_root
    )
    futures = pd.read_csv(path)
    futures["bloomberg_symbol"] = futures["bloomberg_symbol"].astype(str).str.strip()
    futures["index"] = futures["index"].astype(str).str.strip()
    futures["currency"] = futures["currency"].apply(normalize_currency).fillna("USD")
    futures = futures[(futures["bloomberg_symbol"] != "") & (futures["index"] != "")]

    future_to_index = (
        futures[["bloomberg_symbol", "index"]]
        .drop_duplicates(subset=["bloomberg_symbol"], keep="first")
        .set_index("bloomberg_symbol")["index"]
        .to_dict()
    )
    index_to_currency = (
        futures[["index", "currency"]]
        .drop_duplicates(subset=["index"], keep="first")
        .set_index("index")["currency"]
        .to_dict()
    )
    return future_to_index, index_to_currency


def load_index_currency_mapping(
    futures_symbols_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    _, index_to_currency = load_index_reference_mappings(
        futures_symbols_path=futures_symbols_path,
        repo_root=repo_root,
    )
    return index_to_currency


def load_fx_minute(
    currency: str,
    fx_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> pd.Series:
    base_dir = Path(fx_dir) if fx_dir is not None else resolve_repo_path(
        "data", "raw", "currencies", "minute_bars", repo_root=repo_root
    )
    fx_file = base_dir / f"{currency}USD_full_1min.txt"
    fx_df = pd.read_csv(
        fx_file,
        header=None,
        index_col=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={"date": "string", "time": "string"},
    )
    fx_df["date"] = fx_df["date"].str.zfill(8)
    fx_df["time"] = fx_df["time"].str.zfill(6)
    fx_df["timestamp"] = pd.to_datetime(
        fx_df["date"].astype(str) + " " + fx_df["time"].astype(str)
    ).dt.tz_localize("America/New_York")
    return fx_df.set_index("timestamp")["close"]


def compute_exchange_close_times(
    exchange_mic: str,
    offset_str: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    cal = mcal.get_calendar(exchange_mic)
    sched = cal.schedule(start_date=start_date, end_date=end_date)
    close_times_local = sched["market_close"].dt.tz_convert(str(cal.tz))
    close_times_only_local = close_times_local.dt.time
    most_common_local_close = close_times_only_local.mode()[0]
    is_normal_close = close_times_only_local == most_common_local_close

    close_times_et = close_times_local[is_normal_close].dt.tz_convert("America/New_York")
    return close_times_et + pd.Timedelta(offset_str)


def compute_fx_daily_at_close(
    fx_minute: pd.Series,
    close_times: pd.Series,
) -> pd.Series:
    fx_idx_int = fx_minute.index.values.astype("int64")
    close_times_int = close_times.values.astype("int64")
    indices = np.searchsorted(fx_idx_int, close_times_int, side="right") - 1
    valid = indices >= 0
    fx_at_close = pd.Series(
        data=fx_minute.values[indices[valid]],
        index=close_times.index[valid],
        dtype=float,
    )
    return fx_at_close.pct_change()


def convert_returns_to_usd(
    native_returns: pd.Series | pd.DataFrame,
    fx_returns: pd.Series,
) -> pd.Series | pd.DataFrame:
    if isinstance(native_returns, pd.DataFrame):
        common_dates = native_returns.index.intersection(fx_returns.index)
        fx_aligned = fx_returns.loc[common_dates]
        native_aligned = native_returns.loc[common_dates]
        return (1 + native_aligned).multiply(1 + fx_aligned, axis=0) - 1

    common_dates = native_returns.index.intersection(fx_returns.index)
    fx_aligned = fx_returns.loc[common_dates]
    native_aligned = native_returns.loc[common_dates]
    return (1 + native_aligned) * (1 + fx_aligned) - 1


__all__ = [
    "compute_exchange_close_times",
    "compute_fx_daily_at_close",
    "convert_returns_to_usd",
    "is_usd_currency",
    "load_fx_minute",
    "load_index_currency_mapping",
    "load_index_reference_mappings",
    "normalize_currency",
]
