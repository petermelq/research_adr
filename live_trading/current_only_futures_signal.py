#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import blpapi
import pandas as pd
import pandas_market_calendars as mcal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adr_strategy_kernel.common import load_adr_info, load_close_offsets, ASIA_EXCHANGES

DEFAULT_BETAS = REPO_ROOT / "data" / "processed" / "models" / "ordinary_betas_index_only.csv"
DEFAULT_FUTURES_SYMBOLS = REPO_ROOT / "data" / "raw" / "futures_symbols.csv"
DEFAULT_CLOSE_OFFSETS = REPO_ROOT / "data" / "raw" / "close_time_offsets.csv"
DEFAULT_CACHE_ROOT = REPO_ROOT / "live_trading" / "cache" / "current_only_futures_signal"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "live_trading" / "output"
NY_TZ = "America/New_York"
EUROPE_REGION = "EUROPE"
ASIA_REGION = "ASIA"
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
    parser.add_argument("--blpapi-host", default="localhost")
    parser.add_argument("--blpapi-port", type=int, default=8194)
    parser.add_argument("--subscription-timeout", type=float, default=15.0, help="Seconds to wait for futures prices via subscription.")
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


def _fetch_intraday_bars_blpapi(
    ticker: str,
    trade_date: pd.Timestamp,
    event_type: str,
    timeout_ms: int,
    server_host: str,
    server_port: int,
) -> pd.DataFrame:
    """Fetch 1-minute intraday bars for one ticker via blpapi IntradayBarRequest."""
    session_opts = blpapi.SessionOptions()
    session_opts.setServerHost(server_host)
    session_opts.setServerPort(server_port)
    session = blpapi.Session(session_opts)

    if not session.start():
        raise RuntimeError("Failed to start blpapi session")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata")

    svc = session.getService("//blp/refdata")
    req = svc.createRequest("IntradayBarRequest")
    req.set("security", ticker)
    req.set("eventType", event_type)
    req.set("interval", 1)

    date_str = trade_date.strftime("%Y-%m-%d")
    start_utc = _dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    end_utc = start_utc + _dt.timedelta(hours=23, minutes=59)
    req.set("startDateTime", start_utc)
    req.set("endDateTime", end_utc)

    session.sendRequest(req)

    bars: list[dict] = []
    deadline = pd.Timestamp.now() + pd.Timedelta(milliseconds=timeout_ms)

    try:
        while True:
            remaining_ms = int((deadline - pd.Timestamp.now()).total_seconds() * 1000)
            if remaining_ms <= 0:
                break
            event = session.nextEvent(remaining_ms)
            for msg in event:
                if not msg.hasElement("barData"):
                    continue
                bar_data = msg.getElement("barData")
                if not bar_data.hasElement("barTickData"):
                    continue
                tick_data = bar_data.getElement("barTickData")
                for i in range(tick_data.numValues()):
                    bar = tick_data.getValue(i)
                    t = bar.getElementAsDatetime("time")
                    c = bar.getElementAsFloat("close")
                    bars.append({"time": t, "close": c})
            if event.eventType() == blpapi.Event.RESPONSE:
                break
    finally:
        session.stop()

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")


def fetch_intraday_price_series(
    ticker: str,
    trade_date: pd.Timestamp,
    cache_file: Path,
    timeout_ms: int,
    bdib_type: str,
    bdib_session: str,
    refresh: bool,
    blpapi_host: str = "localhost",
    blpapi_port: int = 8194,
) -> pd.Series:
    if not refresh:
        cached = read_cached_bars(cache_file)
        if cached is not None:
            return cached

    raw = _fetch_intraday_bars_blpapi(
        ticker=ticker,
        trade_date=trade_date,
        event_type=bdib_type,
        timeout_ms=timeout_ms,
        server_host=blpapi_host,
        server_port=blpapi_port,
    )
    normalized = normalize_bdib_frame(raw)
    if normalized.empty:
        raise ValueError(f"Bloomberg returned no intraday data for {ticker} on {trade_date.date()}")
    price_series = extract_bar_price_series(normalized)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    price_series.to_frame().to_parquet(cache_file)
    return price_series


def fetch_bdp_snapshot(
    tickers: Iterable[str],
    field: str,
    timeout_ms: int,
    server_host: str = "localhost",
    server_port: int = 8194,
) -> pd.Series:
    """Fetch a single reference-data field for a list of tickers via direct blpapi."""
    unique = list(dict.fromkeys([t for t in tickers if isinstance(t, str) and t]))
    if not unique:
        return pd.Series(dtype=object if field == FIELD_FUT_CUR_GEN_TICKER else float)

    session_opts = blpapi.SessionOptions()
    session_opts.setServerHost(server_host)
    session_opts.setServerPort(server_port)
    session = blpapi.Session(session_opts)

    if not session.start():
        raise RuntimeError("Failed to start blpapi session")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata service")

    svc = session.getService("//blp/refdata")
    req = svc.createRequest("ReferenceDataRequest")
    for t in unique:
        req.getElement("securities").appendValue(t)
    req.getElement("fields").appendValue(field)
    session.sendRequest(req)

    results: dict[str, object] = {}
    timeout_s = timeout_ms / 1000.0
    deadline = pd.Timestamp.now() + pd.Timedelta(seconds=timeout_s)

    try:
        while True:
            remaining_ms = int((deadline - pd.Timestamp.now()).total_seconds() * 1000)
            if remaining_ms <= 0:
                break
            event = session.nextEvent(remaining_ms)
            for msg in event:
                if not msg.hasElement("securityData"):
                    continue
                sec_data = msg.getElement("securityData")
                for i in range(sec_data.numValues()):
                    sec = sec_data.getValue(i)
                    ticker = sec.getElementAsString("security").strip()
                    field_data = sec.getElement("fieldData")
                    if field_data.hasElement(field):
                        results[ticker] = field_data.getElementValue(field)
            if event.eventType() == blpapi.Event.RESPONSE:
                break
    finally:
        session.stop()

    if not results:
        raise ValueError(f"Bloomberg returned no data for field {field}, tickers: {unique}")

    series = pd.Series(results)
    series.index = series.index.astype(str).str.strip()
    if field == FIELD_FUT_CUR_GEN_TICKER:
        return series.astype(str).str.strip()
    return pd.to_numeric(series, errors="coerce")


def fetch_futures_last_price_subscription(
    tickers: list[str],
    timeout_seconds: float = 15.0,
    server_host: str = "localhost",
    server_port: int = 8194,
) -> pd.Series:
    """Fetch LAST_PRICE for futures tickers using a blpapi market data subscription."""
    if not tickers:
        return pd.Series(dtype=float)

    session_opts = blpapi.SessionOptions()
    session_opts.setServerHost(server_host)
    session_opts.setServerPort(server_port)
    session = blpapi.Session(session_opts)

    if not session.start():
        raise RuntimeError("Failed to start blpapi session")
    if not session.openService("//blp/mktdata"):
        raise RuntimeError("Failed to open //blp/mktdata service")

    unique_tickers = list(dict.fromkeys(tickers))
    corr_to_ticker: dict[int, str] = {}
    subs = blpapi.SubscriptionList()
    for i, ticker in enumerate(unique_tickers):
        corr_id = blpapi.CorrelationId(i)
        corr_to_ticker[i] = ticker
        subs.add(ticker, ["LAST_PRICE"], correlationId=corr_id)

    session.subscribe(subs)

    prices: dict[str, float] = {}
    deadline = pd.Timestamp.now() + pd.Timedelta(seconds=timeout_seconds)

    try:
        while len(prices) < len(unique_tickers):
            remaining_ms = int((deadline - pd.Timestamp.now()).total_seconds() * 1000)
            if remaining_ms <= 0:
                break
            event = session.nextEvent(remaining_ms)
            if event.eventType() == blpapi.Event.SUBSCRIPTION_DATA:
                for msg in event:
                    corr_val = msg.correlationIds()[0].value()
                    ticker = corr_to_ticker.get(corr_val)
                    if ticker and msg.hasElement("LAST_PRICE"):
                        val = msg.getElementAsFloat("LAST_PRICE")
                        if not pd.isna(val):
                            prices[ticker] = val
    finally:
        session.unsubscribe(subs)
        session.stop()

    missing = set(unique_tickers) - set(prices)
    if missing:
        print(f"Warning: no LAST_PRICE received within timeout for: {sorted(missing)}", file=sys.stderr)

    return pd.Series(prices, dtype=float)


def load_latest_betas(path: Path) -> pd.Series:
    betas = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    latest = betas.iloc[-1]
    latest.name = betas.index[-1]
    return pd.to_numeric(latest, errors="coerce")


def load_futures_map(path: Path) -> pd.DataFrame:
    futures_map = pd.read_csv(path).copy()
    # Preserve raw symbol (may have trailing space, e.g. "Z ") for building generic tickers.
    # Bloomberg generic ticker format: <raw_symbol>1 Index  (e.g. "Z 1 Index", "VG1 Index")
    futures_map["bloomberg_symbol_raw"] = futures_map["bloomberg_symbol"].astype(str)
    futures_map["bloomberg_symbol"] = futures_map["bloomberg_symbol_raw"].str.strip()
    futures_map["currency"] = futures_map["currency"].astype(str).str.strip().replace({"GBp": "GBP"})
    return futures_map


def build_universe(
    adr_info_path: Path | None,
    betas: pd.Series,
    futures_symbols_path: Path,
) -> pd.DataFrame:
    adr_info = load_adr_info(adr_info_path=adr_info_path)
    futures_map = load_futures_map(futures_symbols_path)

    def _merge_futures(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["currency"] = df["currency"].astype(str).str.strip().replace({"GBp": "GBP"})
        merged = df.merge(
            futures_map[["bloomberg_symbol", "bloomberg_symbol_raw", "currency"]],
            left_on="index_future_bbg",
            right_on="bloomberg_symbol",
            how="left",
            suffixes=("", "_future"),
        )
        merged = merged.rename(columns={"currency_future": "future_currency"})
        merged["future_generic"] = merged["bloomberg_symbol_raw"].astype(str) + "1 Index"
        merged["adr_bbg"] = merged["adr_ticker"] + " US Equity"
        merged["fx_ticker"] = merged["future_currency"].map(lambda cur: None if cur == "USD" else f"{cur}USD Curncy")
        merged["beta"] = merged["adr_ticker"].map(betas)
        return merged.dropna(subset=["beta", "future_currency"])

    # European universe
    eu_info = adr_info[adr_info["region"] == EUROPE_REGION].copy()
    eu_merged = _merge_futures(eu_info)
    eu_merged["baseline_source"] = "adr_intraday"
    eu_merged["sh_per_adr"] = float("nan")
    eu_merged["ordinary_bbg"] = None
    eu_merged["ordinary_fx_ticker"] = None

    # Asian universe
    asia_info = adr_info[adr_info["region"] == ASIA_REGION].copy()
    asia_merged = _merge_futures(asia_info)
    asia_merged["baseline_source"] = "ordinary_close"
    asia_merged["sh_per_adr"] = pd.to_numeric(asia_merged["sh_per_adr"], errors="coerce")
    asia_merged["ordinary_bbg"] = asia_merged["id"].astype(str)
    # ordinary_fx_ticker uses the ordinary currency (not futures currency)
    asia_merged["ordinary_fx_ticker"] = asia_merged["currency"].map(
        lambda cur: None if cur == "USD" else f"{cur}USD Curncy"
    )

    shared_cols = [
        "adr_ticker",
        "adr_bbg",
        "exchange",
        "currency",
        "future_generic",
        "future_currency",
        "fx_ticker",
        "beta",
        "baseline_source",
        "sh_per_adr",
        "ordinary_bbg",
        "ordinary_fx_ticker",
    ]
    combined = pd.concat([eu_merged[shared_cols], asia_merged[shared_cols]], ignore_index=True)
    return combined.sort_values("adr_ticker")


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
    blpapi_host: str = "localhost",
    blpapi_port: int = 8194,
) -> pd.Series:
    cache_file = cache_day_dir / "active_contracts.json"
    if cache_file.exists() and not refresh:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return pd.Series(cached, dtype=object)

    active_contracts = fetch_bdp_snapshot(
        future_generics, FIELD_FUT_CUR_GEN_TICKER, timeout_ms=timeout_ms,
        server_host=blpapi_host, server_port=blpapi_port,
    )
    active_contracts = active_contracts.replace({"": pd.NA}).dropna()
    # FUT_CUR_GEN_TICKER returns e.g. "Z H6"; append " Index" for full BBG ticker
    active_contracts = active_contracts.str.strip() + " Index"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(active_contracts.to_json(), encoding="utf-8")
    return active_contracts


def load_or_fetch_ordinary_prices(
    ordinary_tickers: list[str],
    cache_day_dir: Path,
    timeout_ms: int,
    refresh: bool,
    blpapi_host: str = "localhost",
    blpapi_port: int = 8194,
) -> pd.Series:
    """Fetch PX_LAST for ordinary (foreign) stocks via BDP, with day-level caching."""
    cache_file = cache_day_dir / "ordinary_prices.json"
    if cache_file.exists() and not refresh:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return pd.to_numeric(pd.Series(cached), errors="coerce")

    prices = fetch_bdp_snapshot(
        ordinary_tickers, FIELD_PX_LAST, timeout_ms=timeout_ms,
        server_host=blpapi_host, server_port=blpapi_port,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(prices.to_json(), encoding="utf-8")
    return prices


def compute_day_baselines(
    universe: pd.DataFrame,
    trade_date: pd.Timestamp,
    close_offsets: dict[str, pd.Timedelta],
    cache_day_dir: Path,
    timeout_ms: int,
    bdib_type: str,
    bdib_session: str,
    refresh: bool,
    blpapi_host: str = "localhost",
    blpapi_port: int = 8194,
) -> pd.DataFrame:
    active_contracts = load_or_fetch_active_contracts(
        universe["future_generic"].unique().tolist(),
        cache_day_dir=cache_day_dir,
        timeout_ms=timeout_ms,
        refresh=refresh,
        blpapi_host=blpapi_host,
        blpapi_port=blpapi_port,
    )
    if active_contracts.empty:
        raise ValueError("No active futures contracts resolved from Bloomberg")

    universe = universe.copy()
    universe["future_bbg"] = universe["future_generic"].map(active_contracts)
    universe = universe.dropna(subset=["future_bbg"])

    # Split universe by baseline source
    eu_universe = universe[universe["baseline_source"] == "adr_intraday"]
    asia_universe = universe[universe["baseline_source"] == "ordinary_close"]

    # Build cache paths — only fetch ADR bars for European ADRs
    adr_cache = build_cache_paths(cache_day_dir, "adr_bars", eu_universe["adr_bbg"].unique().tolist())
    fut_cache = build_cache_paths(cache_day_dir, "futures_bars", universe["future_bbg"].unique().tolist())

    # Collect all FX tickers: futures FX for all + ordinary FX for Asian ADRs
    all_fx_tickers = set(universe["fx_ticker"].dropna().unique().tolist())
    all_fx_tickers.update(asia_universe["ordinary_fx_ticker"].dropna().unique().tolist())
    fx_cache = build_cache_paths(cache_day_dir, "fx_bars", list(all_fx_tickers))

    _intraday_kwargs = dict(
        trade_date=trade_date,
        timeout_ms=timeout_ms,
        bdib_type=bdib_type,
        bdib_session=bdib_session,
        refresh=refresh,
        blpapi_host=blpapi_host,
        blpapi_port=blpapi_port,
    )
    adr_bars: dict[str, pd.Series] = {}
    for ticker, cache_file in adr_cache.items():
        try:
            adr_bars[ticker] = fetch_intraday_price_series(ticker=ticker, cache_file=cache_file, **_intraday_kwargs)
        except Exception as exc:
            print(f"Warning: skipping ADR bars for {ticker}: {exc}", file=sys.stderr)

    fut_bars: dict[str, pd.Series] = {}
    for ticker, cache_file in fut_cache.items():
        try:
            fut_bars[ticker] = fetch_intraday_price_series(ticker=ticker, cache_file=cache_file, **_intraday_kwargs)
        except Exception as exc:
            print(f"Warning: skipping futures bars for {ticker}: {exc}", file=sys.stderr)

    fx_bars: dict[str, pd.Series] = {}
    for ticker, cache_file in fx_cache.items():
        try:
            fx_bars[ticker] = fetch_intraday_price_series(ticker=ticker, cache_file=cache_file, **_intraday_kwargs)
        except Exception as exc:
            print(f"Warning: skipping FX bars for {ticker}: {exc}", file=sys.stderr)

    # Fetch ordinary prices for Asian ADRs (cached at day level)
    ordinary_prices: pd.Series = pd.Series(dtype=float)
    if not asia_universe.empty:
        ordinary_tickers = asia_universe["ordinary_bbg"].dropna().unique().tolist()
        ordinary_prices = load_or_fetch_ordinary_prices(
            ordinary_tickers,
            cache_day_dir=cache_day_dir,
            timeout_ms=timeout_ms,
            refresh=refresh,
            blpapi_host=blpapi_host,
            blpapi_port=blpapi_port,
        )

    rows: list[dict[str, object]] = []
    for row in universe.itertuples(index=False):
        offset = close_offsets.get(row.exchange)
        if offset is None:
            continue

        close_time = compute_close_time(row.exchange, trade_date, offset)
        if close_time is None or ny_now() < close_time:
            continue

        if row.future_bbg not in fut_bars:
            print(f"Warning: no futures bars for {row.future_bbg}, skipping {row.adr_ticker}", file=sys.stderr)
            continue
        baseline_fut = select_price_at_or_before(fut_bars[row.future_bbg], close_time)
        baseline_fx = 1.0
        if isinstance(row.fx_ticker, str) and row.fx_ticker:
            if row.fx_ticker not in fx_bars:
                print(f"Warning: no FX bars for {row.fx_ticker}, skipping {row.adr_ticker}", file=sys.stderr)
                continue
            baseline_fx = select_price_at_or_before(fx_bars[row.fx_ticker], close_time)

        if row.baseline_source == "adr_intraday":
            # European: use ADR intraday price at close time
            if row.adr_bbg not in adr_bars:
                print(f"Warning: no ADR bars for {row.adr_bbg}, skipping {row.adr_ticker}", file=sys.stderr)
                continue
            baseline_adr = select_price_at_or_before(adr_bars[row.adr_bbg], close_time)
        else:
            # Asian: use ordinary closing price converted to USD via ADR ratio
            ordinary_px_last = ordinary_prices.get(row.ordinary_bbg)
            if ordinary_px_last is None or pd.isna(ordinary_px_last):
                print(f"Warning: no ordinary price for {row.ordinary_bbg}, skipping {row.adr_ticker}", file=sys.stderr)
                continue
            ordinary_fx = 1.0
            if isinstance(row.ordinary_fx_ticker, str) and row.ordinary_fx_ticker:
                if row.ordinary_fx_ticker not in fx_bars:
                    print(f"Warning: no ordinary FX bars for {row.ordinary_fx_ticker}, skipping {row.adr_ticker}", file=sys.stderr)
                    continue
                ordinary_fx = select_price_at_or_before(fx_bars[row.ordinary_fx_ticker], close_time)
            baseline_adr = float(ordinary_px_last) * float(row.sh_per_adr) * ordinary_fx

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
    blpapi_host: str = "localhost",
    blpapi_port: int = 8194,
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
        blpapi_host=blpapi_host,
        blpapi_port=blpapi_port,
    )


def compute_live_signals(
    baseline_df: pd.DataFrame,
    timeout_ms: int,
    blpapi_host: str = "localhost",
    blpapi_port: int = 8194,
    subscription_timeout: float = 15.0,
) -> pd.DataFrame:
    current_adr = fetch_bdp_snapshot(
        baseline_df["adr_bbg"].tolist(), FIELD_PX_LAST, timeout_ms=timeout_ms,
        server_host=blpapi_host, server_port=blpapi_port,
    )
    current_fut = fetch_futures_last_price_subscription(
        baseline_df["future_bbg"].tolist(),
        timeout_seconds=subscription_timeout,
        server_host=blpapi_host,
        server_port=blpapi_port,
    )
    fx_tickers = baseline_df["fx_ticker"].dropna().unique().tolist()
    current_fx = (
        fetch_bdp_snapshot(fx_tickers, FIELD_PX_LAST, timeout_ms=timeout_ms,
                           server_host=blpapi_host, server_port=blpapi_port)
        if fx_tickers else pd.Series(dtype=float)
    )

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
        blpapi_host=args.blpapi_host,
        blpapi_port=args.blpapi_port,
    )
    output = compute_live_signals(
        baseline_df=baseline_df,
        timeout_ms=args.timeout_ms,
        blpapi_host=args.blpapi_host,
        blpapi_port=args.blpapi_port,
        subscription_timeout=args.subscription_timeout,
    )
    dated_file, latest_file = save_outputs(output, args.output_dir, trade_date)
    #output = output.merge(latest_betas.rename('beta'), left_on='adr_ticker', right_index=True)
    #import IPython; IPython.embed()
    print(f"Saved {len(output)} live signals to {dated_file}")
    print(f"Updated latest snapshot at {latest_file}")
    #print(output[["adr_ticker", "future_return", "adr_return", "beta","signal"]].sort_values('signal').to_string(index=False))
    print('HK ONLY')
    print('')
    hk_tickers = universe[universe['exchange']=='XHKG']['adr_ticker'].tolist()
    print(output[output['adr_ticker'].isin(hk_tickers)][["adr_ticker", "future_return", "adr_return", "current_adr", "beta","signal"]].sort_values('signal').to_string(index=False))
    

if __name__ == "__main__":
    main()
