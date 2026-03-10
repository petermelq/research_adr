#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import pandas_market_calendars as mcal
from linux_xbbg import blp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adr_strategy_kernel.common import load_adr_info, load_close_offsets

DEFAULT_BETAS = REPO_ROOT / "data" / "processed" / "models" / "ordinary_betas_index_only.csv"
DEFAULT_FUTURES_SYMBOLS = REPO_ROOT / "data" / "raw" / "futures_symbols.csv"
DEFAULT_CLOSE_OFFSETS = REPO_ROOT / "data" / "raw" / "close_time_offsets.csv"
DEFAULT_CACHE_ROOT = REPO_ROOT / "live_trading" / "cache" / "current_only_futures_signal"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "live_trading" / "output"
NY_TZ = "America/New_York"
EUROPE_REGION = "EUROPE"
FIELD_PX_LAST = "PX_LAST"
FIELD_FUT_CUR_GEN_TICKER = "FUT_CUR_GEN_TICKER"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a live only-futures ADR signal for European ADRs using cached "
            "same-day Bloomberg intraday bars and fresh Bloomberg spot prices."
        )
    )
    parser.add_argument("--trade-date", type=str, default=None, help="Trading date in YYYY-MM-DD. Defaults to today in New York.")
    parser.add_argument("--adr-info", type=Path, default=None)
    parser.add_argument("--betas", type=Path, default=DEFAULT_BETAS)
    parser.add_argument("--futures-symbols", type=Path, default=DEFAULT_FUTURES_SYMBOLS)
    parser.add_argument("--close-offsets", type=Path, default=DEFAULT_CLOSE_OFFSETS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-day-cache", action="store_true", help="Redownload same-day intraday bars instead of reusing cache.")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--bdib-session", default="day")
    parser.add_argument("--bdib-type", default="TRADE")
    return parser.parse_args()


def ny_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ)


def parse_trade_date(trade_date: str | None) -> pd.Timestamp:
    if trade_date is None:
        return ny_now().normalize()
    return pd.Timestamp(trade_date).tz_localize(NY_TZ).normalize()


def sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def normalize_bdib_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if out.columns.nlevels > 1:
            out.columns = out.columns.get_level_values(-1)
        else:
            out.columns = out.columns.to_flat_index()
    out.columns = [str(col).strip().lower() for col in out.columns]
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize(NY_TZ)
    else:
        out.index = out.index.tz_convert(NY_TZ)
    return out.sort_index()


def extract_bar_price_series(df: pd.DataFrame) -> pd.Series:
    for column in ("close", "last_price", "px_last", "value", "price"):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").rename("price")
    raise ValueError(f"Unable to find a price column in intraday Bloomberg data. Columns: {list(df.columns)}")


def read_cached_bars(cache_file: Path) -> pd.Series | None:
    if not cache_file.exists():
        return None
    cached = pd.read_parquet(cache_file)
    if "price" not in cached.columns:
        raise ValueError(f"Cached file missing price column: {cache_file}")
    cached.index = pd.DatetimeIndex(cached.index)
    if cached.index.tz is None:
        cached.index = cached.index.tz_localize(NY_TZ)
    else:
        cached.index = cached.index.tz_convert(NY_TZ)
    return pd.to_numeric(cached["price"], errors="coerce").rename("price").sort_index()


def fetch_intraday_price_series(
    ticker: str,
    trade_date: pd.Timestamp,
    cache_file: Path,
    timeout_ms: int,
    bdib_type: str,
    bdib_session: str,
    refresh: bool,
) -> pd.Series:
    if not refresh:
        cached = read_cached_bars(cache_file)
        if cached is not None:
            return cached

    raw = blp.bdib(
        ticker,
        dt=trade_date.strftime("%Y-%m-%d"),
        typ=bdib_type,
        timeout=timeout_ms,
        session=bdib_session,
    )
    normalized = normalize_bdib_frame(raw)
    if normalized.empty:
        raise ValueError(f"Bloomberg returned no intraday data for {ticker} on {trade_date.date()}")
    price_series = extract_bar_price_series(normalized)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    price_series.to_frame().to_parquet(cache_file)
    return price_series


def fetch_bdp_snapshot(tickers: Iterable[str], field: str, timeout_ms: int) -> pd.Series:
    unique = list(dict.fromkeys([ticker for ticker in tickers if isinstance(ticker, str) and ticker]))
    if not unique:
        return pd.Series(dtype=object if field == FIELD_FUT_CUR_GEN_TICKER else float)

    df = blp.bdp(unique, [field], timeout=timeout_ms)
    if df.empty:
        raise ValueError(f"Bloomberg returned no BDP data for field {field}")

    out = df.copy()
    out.index = out.index.astype(str).str.strip()
    normalized_columns = {str(col).strip().upper(): col for col in out.columns}
    if field not in normalized_columns:
        raise ValueError(f"Field {field} missing from BDP response. Columns: {list(out.columns)}")

    series = out[normalized_columns[field]]
    series.index = out.index
    if field == FIELD_FUT_CUR_GEN_TICKER:
        return series.astype(str).str.strip()
    return pd.to_numeric(series, errors="coerce")


def load_latest_betas(path: Path) -> pd.Series:
    betas = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    latest = betas.iloc[-1]
    latest.name = betas.index[-1]
    return pd.to_numeric(latest, errors="coerce")


def load_futures_map(path: Path) -> pd.DataFrame:
    futures_map = pd.read_csv(path).copy()
    futures_map["bloomberg_symbol"] = futures_map["bloomberg_symbol"].astype(str).str.strip()
    futures_map["currency"] = futures_map["currency"].astype(str).str.strip().replace({"GBp": "GBP"})
    return futures_map


def build_universe(
    adr_info_path: Path | None,
    betas: pd.Series,
    futures_symbols_path: Path,
) -> pd.DataFrame:
    adr_info = load_adr_info(adr_info_path=adr_info_path)
    adr_info = adr_info[adr_info["region"] == EUROPE_REGION].copy()
    adr_info["currency"] = adr_info["currency"].astype(str).str.strip().replace({"GBp": "GBP"})

    futures_map = load_futures_map(futures_symbols_path)
    merged = adr_info.merge(
        futures_map[["bloomberg_symbol", "currency"]],
        left_on="index_future_bbg",
        right_on="bloomberg_symbol",
        how="left",
        suffixes=("", "_future"),
    )
    merged = merged.rename(columns={"currency_future": "future_currency"})
    merged["future_generic"] = merged["index_future_bbg"].astype(str).str.strip() + " 1 Index"
    merged["adr_bbg"] = merged["adr_ticker"] + " US Equity"
    merged["fx_ticker"] = merged["future_currency"].map(lambda cur: None if cur == "USD" else f"{cur}USD Curncy")
    merged["beta"] = merged["adr_ticker"].map(betas)
    merged = merged.dropna(subset=["beta", "future_currency"])
    return merged[
        [
            "adr_ticker",
            "adr_bbg",
            "exchange",
            "currency",
            "future_generic",
            "future_currency",
            "fx_ticker",
            "beta",
        ]
    ].sort_values("adr_ticker")


def compute_close_time(exchange: str, trade_date: pd.Timestamp, offset: pd.Timedelta) -> pd.Timestamp | None:
    schedule = mcal.get_calendar(exchange).schedule(
        start_date=trade_date.strftime("%Y-%m-%d"),
        end_date=trade_date.strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        return None
    close_time = schedule["market_close"].iloc[0]
    return close_time.tz_convert(NY_TZ) + offset


def select_price_at_or_before(series: pd.Series, cutoff: pd.Timestamp) -> float:
    eligible = series[series.index <= cutoff]
    if eligible.empty:
        raise ValueError(f"No intraday data at or before {cutoff}")
    return float(eligible.iloc[-1])


def build_cache_paths(cache_day_dir: Path, key: str, tickers: Iterable[str]) -> dict[str, Path]:
    return {
        ticker: cache_day_dir / key / f"{sanitize_for_filename(ticker)}.parquet"
        for ticker in tickers
    }


def load_or_fetch_active_contracts(
    future_generics: Iterable[str],
    cache_day_dir: Path,
    timeout_ms: int,
    refresh: bool,
) -> pd.Series:
    cache_file = cache_day_dir / "active_contracts.json"
    if cache_file.exists() and not refresh:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return pd.Series(cached, dtype=object)

    active_contracts = fetch_bdp_snapshot(future_generics, FIELD_FUT_CUR_GEN_TICKER, timeout_ms=timeout_ms)
    active_contracts = active_contracts.replace({"": pd.NA}).dropna()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(active_contracts.to_json(), encoding="utf-8")
    return active_contracts


def compute_day_baselines(
    universe: pd.DataFrame,
    trade_date: pd.Timestamp,
    close_offsets: dict[str, pd.Timedelta],
    cache_day_dir: Path,
    timeout_ms: int,
    bdib_type: str,
    bdib_session: str,
    refresh: bool,
) -> pd.DataFrame:
    active_contracts = load_or_fetch_active_contracts(
        universe["future_generic"].unique().tolist(),
        cache_day_dir=cache_day_dir,
        timeout_ms=timeout_ms,
        refresh=refresh,
    )
    if active_contracts.empty:
        raise ValueError("No active futures contracts resolved from Bloomberg")

    universe = universe.copy()
    universe["future_bbg"] = universe["future_generic"].map(active_contracts)
    universe = universe.dropna(subset=["future_bbg"])

    adr_cache = build_cache_paths(cache_day_dir, "adr_bars", universe["adr_bbg"].unique().tolist())
    fut_cache = build_cache_paths(cache_day_dir, "futures_bars", universe["future_bbg"].unique().tolist())
    fx_cache = build_cache_paths(cache_day_dir, "fx_bars", universe["fx_ticker"].dropna().unique().tolist())

    adr_bars = {
        ticker: fetch_intraday_price_series(
            ticker=ticker,
            trade_date=trade_date,
            cache_file=cache_file,
            timeout_ms=timeout_ms,
            bdib_type=bdib_type,
            bdib_session=bdib_session,
            refresh=refresh,
        )
        for ticker, cache_file in adr_cache.items()
    }
    fut_bars = {
        ticker: fetch_intraday_price_series(
            ticker=ticker,
            trade_date=trade_date,
            cache_file=cache_file,
            timeout_ms=timeout_ms,
            bdib_type=bdib_type,
            bdib_session=bdib_session,
            refresh=refresh,
        )
        for ticker, cache_file in fut_cache.items()
    }
    fx_bars = {
        ticker: fetch_intraday_price_series(
            ticker=ticker,
            trade_date=trade_date,
            cache_file=cache_file,
            timeout_ms=timeout_ms,
            bdib_type=bdib_type,
            bdib_session=bdib_session,
            refresh=refresh,
        )
        for ticker, cache_file in fx_cache.items()
    }

    rows: list[dict[str, object]] = []
    for row in universe.itertuples(index=False):
        offset = close_offsets.get(row.exchange)
        if offset is None:
            continue

        close_time = compute_close_time(row.exchange, trade_date, offset)
        if close_time is None or ny_now() < close_time:
            continue

        baseline_adr = select_price_at_or_before(adr_bars[row.adr_bbg], close_time)
        baseline_fut = select_price_at_or_before(fut_bars[row.future_bbg], close_time)
        baseline_fx = 1.0
        if isinstance(row.fx_ticker, str) and row.fx_ticker:
            baseline_fx = select_price_at_or_before(fx_bars[row.fx_ticker], close_time)

        rows.append(
            {
                "adr_ticker": row.adr_ticker,
                "adr_bbg": row.adr_bbg,
                "exchange": row.exchange,
                "close_time": close_time,
                "beta": float(row.beta),
                "future_bbg": row.future_bbg,
                "future_currency": row.future_currency,
                "fx_ticker": row.fx_ticker,
                "baseline_adr": baseline_adr,
                "baseline_future_native": baseline_fut,
                "baseline_fx": baseline_fx,
            }
        )

    baseline_df = pd.DataFrame(rows).sort_values("adr_ticker")
    if baseline_df.empty:
        raise ValueError(f"No baselines available for {trade_date.date()}. Check market close status and Bloomberg intraday data.")

    baseline_file = cache_day_dir / "baselines.parquet"
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_parquet(baseline_file, index=False)
    return baseline_df


def load_or_compute_day_baselines(
    universe: pd.DataFrame,
    trade_date: pd.Timestamp,
    close_offsets: dict[str, pd.Timedelta],
    cache_day_dir: Path,
    timeout_ms: int,
    bdib_type: str,
    bdib_session: str,
    refresh: bool,
) -> pd.DataFrame:
    baseline_file = cache_day_dir / "baselines.parquet"
    if baseline_file.exists() and not refresh:
        cached = pd.read_parquet(baseline_file)
        cached["close_time"] = pd.to_datetime(cached["close_time"], utc=True).dt.tz_convert(NY_TZ)
        return cached
    return compute_day_baselines(
        universe=universe,
        trade_date=trade_date,
        close_offsets=close_offsets,
        cache_day_dir=cache_day_dir,
        timeout_ms=timeout_ms,
        bdib_type=bdib_type,
        bdib_session=bdib_session,
        refresh=refresh,
    )


def compute_live_signals(
    baseline_df: pd.DataFrame,
    timeout_ms: int,
) -> pd.DataFrame:
    current_adr = fetch_bdp_snapshot(baseline_df["adr_bbg"].tolist(), FIELD_PX_LAST, timeout_ms=timeout_ms)
    current_fut = fetch_bdp_snapshot(baseline_df["future_bbg"].tolist(), FIELD_PX_LAST, timeout_ms=timeout_ms)
    fx_tickers = baseline_df["fx_ticker"].dropna().unique().tolist()
    current_fx = fetch_bdp_snapshot(fx_tickers, FIELD_PX_LAST, timeout_ms=timeout_ms) if fx_tickers else pd.Series(dtype=float)

    out = baseline_df.copy()
    out["current_adr"] = out["adr_bbg"].map(current_adr)
    out["current_future_native"] = out["future_bbg"].map(current_fut)
    out["current_fx"] = out["fx_ticker"].map(current_fx).fillna(1.0)
    out["future_return"] = (
        (out["current_future_native"] * out["current_fx"])
        / (out["baseline_future_native"] * out["baseline_fx"])
        - 1.0
    )
    out["adr_return"] = out["current_adr"] / out["baseline_adr"] - 1.0
    out["signal"] = out["future_return"] * out["beta"] - out["adr_return"]
    out["asof_time"] = ny_now()
    out["trade_date"] = out["asof_time"].dt.normalize()
    return out[
        [
            "trade_date",
            "asof_time",
            "adr_ticker",
            "exchange",
            "beta",
            "close_time",
            "adr_bbg",
            "baseline_adr",
            "current_adr",
            "future_bbg",
            "future_currency",
            "baseline_future_native",
            "current_future_native",
            "fx_ticker",
            "baseline_fx",
            "current_fx",
            "future_return",
            "adr_return",
            "signal",
        ]
    ].sort_values("signal", ascending=False)


def save_outputs(output: pd.DataFrame, output_dir: Path, trade_date: pd.Timestamp) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dated_file = output_dir / f"current_only_futures_signal_{trade_date.strftime('%Y-%m-%d')}.csv"
    latest_file = output_dir / "current_only_futures_signal_latest.csv"
    output.to_csv(dated_file, index=False)
    output.to_csv(latest_file, index=False)
    return dated_file, latest_file


def main() -> None:
    args = parse_args()
    trade_date = parse_trade_date(args.trade_date)
    cache_day_dir = args.cache_root / trade_date.strftime("%Y-%m-%d")

    latest_betas = load_latest_betas(args.betas)
    universe = build_universe(args.adr_info, latest_betas, args.futures_symbols)
    close_offsets = load_close_offsets(args.close_offsets)
    baseline_df = load_or_compute_day_baselines(
        universe=universe,
        trade_date=trade_date,
        close_offsets=close_offsets,
        cache_day_dir=cache_day_dir,
        timeout_ms=args.timeout_ms,
        bdib_type=args.bdib_type,
        bdib_session=args.bdib_session,
        refresh=args.refresh_day_cache,
    )
    output = compute_live_signals(
        baseline_df=baseline_df,
        timeout_ms=args.timeout_ms,
    )
    dated_file, latest_file = save_outputs(output, args.output_dir, trade_date)

    print(f"Saved {len(output)} live signals to {dated_file}")
    print(f"Updated latest snapshot at {latest_file}")
    print(output[["adr_ticker", "signal", "future_return", "adr_return"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
