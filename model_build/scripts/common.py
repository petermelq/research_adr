from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adr_strategy_kernel.common import normalize_currency, strip_us_equity_suffix
from adr_strategy_kernel.pipelines.closing_domestic_prices import SUPPORTED_FX_CURRENCIES

ARTIFACT_REQUIRED_FILES = {
    "manifest": "manifest.yaml",
    "adr_info": "metadata/adr_info.csv",
    "futures_symbols": "metadata/futures_symbols.csv",
    "close_time_offsets": "metadata/close_time_offsets.csv",
    "bloomberg_close_times": "metadata/bloomberg_close_times.csv",
    "all_splits": "metadata/all_splits.csv",
    "share_reclass": "metadata/share_reclass.csv",
}


def parse_hhmm(value: str) -> tuple[int, int]:
    hours_str, minutes_str = str(value).split(":")
    return int(hours_str), int(minutes_str)


def artifact_paths(artifact_dir: str | Path) -> dict[str, Path]:
    base = Path(artifact_dir)
    paths = {name: base / relative for name, relative in ARTIFACT_REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Artifact is missing required files:\n" + "\n".join(missing))
    return paths


def load_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    manifest_path = artifact_paths(artifact_dir)["manifest"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    required_keys = [
        "trade_date",
        "fixed_trade_time",
        "lookbacks",
        "kernel",
        "sources",
    ]
    missing = [key for key in required_keys if key not in manifest]
    if missing:
        raise KeyError(f"Manifest {manifest_path} is missing keys: {', '.join(missing)}")
    return manifest


def load_artifact_tables(artifact_dir: str | Path) -> dict[str, pd.DataFrame]:
    paths = artifact_paths(artifact_dir)
    adr_info = pd.read_csv(paths["adr_info"]).copy()
    adr_info["adr_ticker"] = strip_us_equity_suffix(adr_info["adr"])
    adr_info["index_future_bbg"] = adr_info["index_future_bbg"].astype(str).str.strip()
    futures_symbols = pd.read_csv(paths["futures_symbols"]).copy()
    futures_symbols["bloomberg_symbol"] = futures_symbols["bloomberg_symbol"].astype(str).str.strip()
    futures_symbols["first_rate_symbol"] = futures_symbols["first_rate_symbol"].astype(str).str.strip()
    futures_symbols["currency"] = futures_symbols["currency"].apply(normalize_currency)

    trade_symbols_path = Path(artifact_dir) / "strategy" / "trade_symbols.csv"
    trade_symbols = pd.read_csv(trade_symbols_path) if trade_symbols_path.exists() else None
    return {
        "adr_info": adr_info,
        "futures_symbols": futures_symbols,
        "trade_symbols": trade_symbols,
    }


def derive_required_inputs(
    adr_info: pd.DataFrame,
    futures_symbols: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    market_etf_tickers = pd.DataFrame(
        {"ticker": sorted(adr_info["market_etf_hedge"].dropna().astype(str).unique().tolist())}
    )
    adr_tickers = pd.DataFrame({"ticker": sorted(adr_info["adr_ticker"].dropna().astype(str).unique().tolist())})

    required_futures = futures_symbols[
        futures_symbols["bloomberg_symbol"].isin(adr_info["index_future_bbg"].dropna().astype(str).str.strip())
    ].drop_duplicates(subset=["first_rate_symbol"]).copy()
    required_futures = required_futures[required_futures["first_rate_symbol"].astype(str) != ""]
    required_futures = required_futures.sort_values("first_rate_symbol").reset_index(drop=True)

    fx_currencies = {
        currency
        for currency in adr_info["currency"].dropna().map(normalize_currency).tolist()
        if currency in SUPPORTED_FX_CURRENCIES and currency != "USD"
    }
    fx_currencies.update(
        {
            currency
            for currency in required_futures["currency"].dropna().map(normalize_currency).tolist()
            if currency in SUPPORTED_FX_CURRENCIES and currency != "USD"
        }
    )
    required_fx = pd.DataFrame({"currency": sorted(fx_currencies)})

    return {
        "adr_tickers": adr_tickers,
        "market_etf_tickers": market_etf_tickers,
        "required_futures": required_futures,
        "required_fx": required_fx,
    }


def _session_start_date(
    calendar_name: str,
    trade_date: str | pd.Timestamp,
    previous_sessions_required: int,
) -> str:
    trade_ts = pd.Timestamp(trade_date).normalize()
    lookback_days = max(365, previous_sessions_required * 3)

    while True:
        probe_start = trade_ts - pd.Timedelta(days=lookback_days)
        schedule = mcal.get_calendar(calendar_name).schedule(
            start_date=probe_start.strftime("%Y-%m-%d"),
            end_date=trade_ts.strftime("%Y-%m-%d"),
        )
        sessions = pd.DatetimeIndex(pd.to_datetime(schedule.index)).normalize()
        sessions = sessions[sessions <= trade_ts]
        needed_sessions = previous_sessions_required + 1
        if len(sessions) >= needed_sessions:
            return sessions[-needed_sessions].strftime("%Y-%m-%d")
        lookback_days *= 2


def derive_start_dates(
    manifest: dict[str, Any],
    adr_info: pd.DataFrame,
) -> dict[str, str]:
    trade_date = pd.Timestamp(manifest["trade_date"]).strftime("%Y-%m-%d")
    lookbacks = manifest["lookbacks"]

    market_previous_sessions = max(
        int(lookbacks["index_beta_days"]) + 1,
        int(lookbacks["hedge_ratio_days"]),
    )
    intraday_previous_sessions = int(lookbacks["covariance_days"])

    exchanges = sorted(adr_info["exchange"].dropna().astype(str).unique().tolist())
    calendars = sorted(set(exchanges + ["XNYS"]))

    market_dates = [
        _session_start_date(calendar_name, trade_date, market_previous_sessions)
        for calendar_name in calendars
    ]
    intraday_dates = [
        _session_start_date(calendar_name, trade_date, intraday_previous_sessions)
        for calendar_name in calendars
    ]
    return {
        "market_data_start_date": min(market_dates),
        "intraday_start_date": min(intraday_dates),
    }


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
