"""
Prepare extended-history Russell residual features for model experiments.

Adds an Asia-specific path:
- target is ADR return from ordinary-theoretical close (USD) to ADR close (USD),
  residualized by existing index-only betas and futures returns
- Russell features use same-day US regular-session open->close returns
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from utils_lasso_residuals import (
    load_ordinary_exchange_mapping,
    load_index_mapping,
    load_index_currency_mapping,
    is_usd_currency,
    normalize_currency,
    compute_aligned_returns,
    residualize_returns,
    get_existing_beta_residuals,
    load_fx_minute,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)

__script_dir__ = Path(__file__).parent.absolute()
ASIA_EXCHANGES = {"XTKS", "XASX", "XHKG", "XSES", "XSHG", "XSHE"}
FEATURE_RECENT_DAYS = 60
FEATURE_MIN_TOTAL_COVERAGE = 0.30
FEATURE_MIN_RECENT_COVERAGE = 0.50
FEATURE_MIN_RECENT_OBS = 20
FEATURE_MIN_COUNT = 20
FEATURE_MIN_SUCCESS_RATIO = 1.0


def load_experiment_universe():
    adr_info = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "adr_info.csv")
    adr_info["adr_ticker"] = adr_info["adr"].str.replace(" US Equity", "", regex=False)
    return set(adr_info["adr_ticker"].dropna().unique().tolist())


def load_ticker_set_from_csv(path, column="ticker"):
    df = pd.read_csv(path)
    if column in df.columns:
        s = df[column]
    elif len(df.columns) == 1:
        s = df[df.columns[0]]
    else:
        raise ValueError(f"Column '{column}' not found in {path} and file has multiple columns.")
    return set(s.dropna().astype(str).str.strip().replace("", np.nan).dropna().tolist())


def resolve_ny_close_target_adrs(exchange_to_tickers, explicit_tickers=None, asia_exchanges=ASIA_EXCHANGES):
    """
    Determine which ADR tickers should use NY-close-to-NY-close target/features.

    Default: ADRs on Asia exchanges.
    Override: if `explicit_tickers` is provided, use exactly that set.
    """
    if explicit_tickers is not None:
        return set(explicit_tickers)
    return {
        adr_ticker
        for ex, ticker_list in exchange_to_tickers.items()
        if ex in asia_exchanges
        for _, adr_ticker, _ in ticker_list
    }


def _daily_us_open_close_returns(russell_ohlcv_dir, tickers, start_date, end_date):
    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    tasks = [(str(russell_ohlcv_dir), t, start_s, end_s) for t in tickers]
    data = {}
    max_workers = min(8, max(1, (os.cpu_count() or 1)))
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for ticker, series in ex.map(_compute_us_open_close_for_ticker, tasks, chunksize=8):
                if ticker is not None and series is not None and not series.empty:
                    data[ticker] = series
    except (PermissionError, OSError) as e:
        print(
            f"ProcessPool unavailable ({e.__class__.__name__}: {e}); "
            f"falling back to ThreadPoolExecutor with {max_workers} workers.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for ticker, series in ex.map(_compute_us_open_close_for_ticker, tasks):
                if ticker is not None and series is not None and not series.empty:
                    data[ticker] = series
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).sort_index()


def _compute_us_open_close_for_ticker(task):
    russell_ohlcv_dir, ticker, start_s, end_s = task
    p = Path(russell_ohlcv_dir) / f"ticker={ticker}" / "data.parquet"
    if not p.exists():
        return None, None
    try:
        # Support both schemas:
        # 1) DatetimeIndex + Close
        # 2) DateTime column + Close
        df = pd.read_parquet(p, columns=["Close"])
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df_dt = pd.read_parquet(p, columns=["DateTime", "Close"])
                if "DateTime" in df_dt.columns:
                    dt = pd.to_datetime(df_dt["DateTime"], errors="coerce")
                    keep = ~dt.isna()
                    df = pd.DataFrame({"Close": df_dt.loc[keep, "Close"].to_numpy()}, index=dt[keep].to_numpy())
            except Exception:
                return None, None
        if not isinstance(df.index, pd.DatetimeIndex):
            return None, None
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
        start_ts = pd.Timestamp(start_s).tz_localize("America/New_York")
        end_ts = pd.Timestamp(end_s).tz_localize("America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty:
            return None, None
        intraday = df.between_time("09:30", "16:00", inclusive="both")
        if intraday.empty:
            return None, None
        daily = intraday["Close"].groupby(intraday.index.normalize()).agg(["first", "last"])
        ret = (daily["last"] / daily["first"] - 1.0).rename(ticker)
        if getattr(ret.index, "tz", None) is not None:
            ret.index = ret.index.tz_localize(None)
        return ticker, ret
    except Exception:
        return None, None


def _daily_us_close_close_returns(russell_ohlcv_dir, tickers, start_date, end_date):
    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    tasks = [(str(russell_ohlcv_dir), t, start_s, end_s) for t in tickers]
    data = {}
    max_workers = min(8, max(1, (os.cpu_count() or 1)))
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for ticker, series in ex.map(_compute_us_ny_close_price_for_ticker, tasks, chunksize=8):
                if ticker is not None and series is not None and not series.empty:
                    data[ticker] = series
    except (PermissionError, OSError) as e:
        print(
            f"ProcessPool unavailable ({e.__class__.__name__}: {e}); "
            f"falling back to ThreadPoolExecutor with {max_workers} workers.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for ticker, series in ex.map(_compute_us_ny_close_price_for_ticker, tasks):
                if ticker is not None and series is not None and not series.empty:
                    data[ticker] = series
    if not data:
        return pd.DataFrame()
    close_px = pd.DataFrame(data).sort_index()
    return close_px.pct_change(fill_method=None)


def _compute_us_ny_close_price_for_ticker(task):
    russell_ohlcv_dir, ticker, start_s, end_s = task
    p = Path(russell_ohlcv_dir) / f"ticker={ticker}" / "data.parquet"
    if not p.exists():
        return None, None
    try:
        df = pd.read_parquet(p, columns=["Close"])
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df_dt = pd.read_parquet(p, columns=["DateTime", "Close"])
                if "DateTime" in df_dt.columns:
                    dt = pd.to_datetime(df_dt["DateTime"], errors="coerce")
                    keep = ~dt.isna()
                    df = pd.DataFrame({"Close": df_dt.loc[keep, "Close"].to_numpy()}, index=dt[keep].to_numpy())
            except Exception:
                return None, None
        if not isinstance(df.index, pd.DatetimeIndex):
            return None, None
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
        start_ts = pd.Timestamp(start_s).tz_localize("America/New_York")
        end_ts = pd.Timestamp(end_s).tz_localize("America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty:
            return None, None
        # Keep only regular NY close rows; early closes are intentionally excluded.
        close_rows = df.between_time("16:00", "16:00", inclusive="both")
        if close_rows.empty:
            return None, None
        series = close_rows["Close"].groupby(close_rows.index.normalize()).last().rename(ticker)
        if getattr(series.index, "tz", None) is not None:
            series.index = series.index.tz_localize(None)
        return ticker, series
    except Exception:
        return None, None


def _daily_futures_domestic_to_ny_return(futures_dir, symbol, exchange_mic, start_date, end_date, exchange_offsets):
    start_ts = pd.Timestamp(start_date, tz="America/New_York")
    fut = pd.read_parquet(
        futures_dir,
        filters=[("symbol", "==", symbol), ("timestamp", ">=", start_ts)],
        columns=["timestamp", "symbol", "close"],
    ).sort_values("timestamp")
    if fut.empty:
        return pd.Series(dtype=float)
    fut["date"] = fut["timestamp"].dt.strftime("%Y-%m-%d")
    fut = fut.set_index("timestamp")

    close_df = (
        mcal.get_calendar(exchange_mic).schedule(start_date=start_date, end_date=end_date)["market_close"]
        .dt.tz_convert("America/New_York")
    )
    close_df.index = close_df.index.astype(str)
    offset = pd.Timedelta(exchange_offsets.get(exchange_mic, "0min"))
    merged_dom = fut.merge(close_df.rename("domestic_close_time"), left_on="date", right_index=True)
    fut_dom = merged_dom.groupby("date")[["domestic_close_time", "close"]].apply(
        lambda x: x[x.index <= x["domestic_close_time"] + offset].iloc[-1]["close"]
        if (x.index <= x["domestic_close_time"] + offset).any()
        else np.nan
    ).rename("fut_domestic_close")

    ny_close = (
        mcal.get_calendar("NYSE").schedule(start_date=start_date, end_date=end_date)["market_close"]
        .dt.tz_convert("America/New_York")
    )
    ny_close.index = ny_close.index.astype(str)
    merged_ny = fut.merge(ny_close.rename("ny_close_time"), left_on="date", right_index=True)
    fut_ny = merged_ny.groupby("date")[["ny_close_time", "close"]].apply(
        lambda x: x[x.index <= x["ny_close_time"]].iloc[-1]["close"]
        if (x.index <= x["ny_close_time"]).any()
        else np.nan
    ).rename("fut_ny_close")

    out = pd.concat([fut_dom, fut_ny], axis=1).dropna()
    if out.empty:
        return pd.Series(dtype=float)
    ret = (out["fut_ny_close"] - out["fut_domestic_close"]) / out["fut_domestic_close"]
    ret.index = pd.to_datetime(ret.index)
    return ret.sort_index()


def _daily_futures_ny_close_to_close_return(futures_dir, symbol, start_date, end_date):
    start_ts = pd.Timestamp(start_date, tz="America/New_York")
    fut = pd.read_parquet(
        futures_dir,
        filters=[("symbol", "==", symbol), ("timestamp", ">=", start_ts)],
        columns=["timestamp", "symbol", "close"],
    ).sort_values("timestamp")
    if fut.empty:
        return pd.Series(dtype=float)

    fut["date"] = fut["timestamp"].dt.strftime("%Y-%m-%d")
    fut = fut.set_index("timestamp")

    ny_close = (
        mcal.get_calendar("NYSE").schedule(start_date=start_date, end_date=end_date)["market_close"]
        .dt.tz_convert("America/New_York")
    )
    ny_close.index = ny_close.index.astype(str)
    merged_ny = fut.merge(ny_close.rename("ny_close_time"), left_on="date", right_index=True)
    fut_ny = merged_ny.groupby("date")[["ny_close_time", "close"]].apply(
        lambda x: x[x.index <= x["ny_close_time"]].iloc[-1]["close"]
        if (x.index <= x["ny_close_time"]).any()
        else np.nan
    ).rename("fut_ny_close")
    fut_ny = fut_ny.dropna()
    if fut_ny.empty:
        return pd.Series(dtype=float)
    ret = fut_ny.pct_change(fill_method=None).dropna()
    ret.index = pd.to_datetime(ret.index)
    return ret.sort_index()


def select_feature_columns_by_coverage(
    russell_residuals,
    common_dates,
    recent_days=FEATURE_RECENT_DAYS,
    min_total_coverage=FEATURE_MIN_TOTAL_COVERAGE,
    min_recent_coverage=FEATURE_MIN_RECENT_COVERAGE,
    min_recent_obs=FEATURE_MIN_RECENT_OBS,
):
    """
    Keep only Russell feature columns with sufficient data coverage.

    This prevents stale/sparse Russell columns from being blindly zero-imputed
    and dominating recent rows with artificial zeros.
    """
    if russell_residuals.empty or len(common_dates) == 0:
        return []
    r_aligned = russell_residuals.loc[common_dates]
    if r_aligned.empty:
        return []

    total_cov = r_aligned.notna().mean(axis=0)
    recent_n = min(int(recent_days), len(r_aligned))
    r_recent = r_aligned.tail(recent_n)
    recent_cov = r_recent.notna().mean(axis=0)
    req_recent_obs = min(int(min_recent_obs), recent_n)
    recent_obs = r_recent.notna().sum(axis=0)

    keep_mask = (
        (total_cov >= float(min_total_coverage)) &
        (recent_cov >= float(min_recent_coverage)) &
        (recent_obs >= int(req_recent_obs))
    )
    return r_aligned.columns[keep_mask].tolist()


def build_feature_matrix(
    ordinary_residuals,
    russell_residuals,
    min_feature_count=FEATURE_MIN_COUNT,
):
    """
    Build final feature matrix with robust Russell feature filtering.

    - Never impute `ordinary_residual` with zero.
    - Drop sparse Russell columns first, then impute residual holes in retained
      Russell columns with 0.0.
    """
    common_dates = ordinary_residuals.index.intersection(russell_residuals.index)
    if len(common_dates) == 0:
        return None

    y = ordinary_residuals.loc[common_dates].astype(float).dropna()
    if y.empty:
        return None
    common_dates = y.index

    keep_cols = select_feature_columns_by_coverage(russell_residuals, common_dates)
    if len(keep_cols) < int(min_feature_count):
        return None

    r_features = russell_residuals.loc[common_dates, keep_cols].copy()
    r_features.columns = [f"russell_{col}" for col in r_features.columns]
    r_features = r_features.fillna(0.0)

    features = pd.concat([y.rename("ordinary_residual"), r_features], axis=1)
    features = features.dropna(subset=["ordinary_residual"])
    if features.empty:
        return None
    return features

def validate_feature_outputs(
    output_dir,
    expected_adrs,
    min_feature_count=FEATURE_MIN_COUNT,
    min_success_ratio=FEATURE_MIN_SUCCESS_RATIO,
    run_written_adrs=None,
):
    """Validate feature completeness and schema; raise on hard failures."""
    output_dir = Path(output_dir)
    feature_files = sorted(output_dir.glob("*.parquet"))
    produced_adrs = {fp.stem for fp in feature_files}
    expected_adrs = set(expected_adrs)

    if not expected_adrs:
        raise RuntimeError("No expected ADR universe for feature validation.")

    success_ratio = len(produced_adrs) / len(expected_adrs)
    missing_adrs = sorted(expected_adrs - produced_adrs)
    schema_errors = []
    audit_rows = []
    for fp in feature_files:
        ticker = fp.stem
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            schema_errors.append((ticker, f"parquet_read_error:{e}"))
            continue
        if "ordinary_residual" not in df.columns:
            schema_errors.append((ticker, "missing_ordinary_residual"))
            continue
        russell_cols = [c for c in df.columns if c.startswith("russell_")]
        if len(russell_cols) < int(min_feature_count):
            schema_errors.append((ticker, f"too_few_russell_features:{len(russell_cols)}"))
        if df["ordinary_residual"].isna().any():
            schema_errors.append((ticker, "ordinary_residual_contains_nan"))
        if not isinstance(df.index, pd.DatetimeIndex):
            schema_errors.append((ticker, "index_not_datetime"))
        else:
            if not df.index.is_monotonic_increasing:
                schema_errors.append((ticker, "index_not_monotonic"))
            if df.index.has_duplicates:
                schema_errors.append((ticker, "index_has_duplicates"))
        audit_rows.append(
            {
                "adr_ticker": ticker,
                "n_rows": int(len(df)),
                "n_russell_features": int(len(russell_cols)),
                "start_date": str(df.index.min()) if len(df) > 0 else "",
                "end_date": str(df.index.max()) if len(df) > 0 else "",
            }
        )

    audit_path = output_dir.parent / f"{output_dir.name}_build_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    if success_ratio < float(min_success_ratio):
        raise RuntimeError(
            f"Feature validation failed: produced={len(produced_adrs)} "
            f"expected={len(expected_adrs)} ratio={success_ratio:.3f} "
            f"< min_success_ratio={float(min_success_ratio):.3f}. "
            f"missing={len(missing_adrs)}"
        )
    if run_written_adrs is not None:
        run_written_adrs = set(run_written_adrs)
        run_ratio = len(run_written_adrs) / len(expected_adrs)
        if run_ratio < float(min_success_ratio):
            missing_run = sorted(expected_adrs - run_written_adrs)
            raise RuntimeError(
                f"Feature validation failed: run_writes={len(run_written_adrs)} "
                f"expected={len(expected_adrs)} ratio={run_ratio:.3f} "
                f"< min_success_ratio={float(min_success_ratio):.3f}. "
                f"missing_run_writes={len(missing_run)}"
            )
    if schema_errors:
        sample = "; ".join([f"{t}:{e}" for t, e in schema_errors[:20]])
        raise RuntimeError(
            f"Feature schema validation failed for {len(schema_errors)} files. "
            f"Sample: {sample}"
        )

    print(
        "Feature validation passed: "
        f"produced={len(produced_adrs)}/{len(expected_adrs)} "
        f"(ratio={success_ratio:.3f}), audit={audit_path}"
    )
    if missing_adrs:
        print("Missing ADR features (first 25): " + ", ".join(missing_adrs[:25]))


def main():
    parser = argparse.ArgumentParser(description="Prepare extended-history Russell residual features.")
    parser.add_argument("--start-date", default=None, help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        default=str(__script_dir__ / ".." / "data" / "processed" / "models" / "with_us_stocks" / "features_extended"),
        help="Output directory for per-ADR feature parquet files.",
    )
    parser.add_argument("--clean-output", action="store_true", help="Delete existing feature parquet files first.")
    parser.add_argument(
        "--min-success-ratio",
        type=float,
        default=FEATURE_MIN_SUCCESS_RATIO,
        help="Minimum produced/expected ADR feature ratio required to pass validation.",
    )
    parser.add_argument(
        "--ny-close-target-tickers",
        default=None,
        help="Comma-separated ADR tickers to use NY-close-to-NY-close mode. Overrides default Asia ADR set.",
    )
    parser.add_argument(
        "--ny-close-target-tickers-file",
        default=None,
        help="CSV file containing ADR tickers to use NY-close-to-NY-close mode. Overrides default Asia ADR set.",
    )
    parser.add_argument(
        "--ny-close-target-tickers-column",
        default="ticker",
        help="Ticker column name for --ny-close-target-tickers-file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for fp in output_dir.glob("*.parquet"):
            fp.unlink()

    params = load_params()
    start_date = args.start_date or params["start_date"]
    end_date = args.end_date or params["end_date"]
    experiment_universe = load_experiment_universe()

    ordinary_to_exchange, ordinary_to_adr = load_ordinary_exchange_mapping(include_asia=True)
    ordinary_to_index, exchange_to_index = load_index_mapping(include_asia=True)
    index_to_currency = load_index_currency_mapping()
    adr_info = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "adr_info.csv")
    adr_info["currency"] = adr_info["currency"].map(normalize_currency)
    adr_info["adr_ticker"] = adr_info["adr"].str.replace(" US Equity", "", regex=False)
    ordinary_to_currency = dict(zip(adr_info["id"], adr_info["currency"]))
    ordinary_to_adr = dict(zip(adr_info["id"], adr_info["adr_ticker"]))

    ordinary_path = __script_dir__ / ".." / "data" / "raw" / "ordinary" / "ord_PX_LAST_adjust_all.csv"
    ordinary_prices = pd.read_csv(ordinary_path, index_col=0, parse_dates=True)

    russell_dir = __script_dir__ / ".." / "data" / "processed" / "russell1000" / "close_at_exchange_auction_adjusted"
    russell_prices_by_exchange = {p.stem: pd.read_csv(p, index_col=0, parse_dates=True) for p in russell_dir.glob("*.csv")}

    aligned_index_prices = pd.read_csv(
        __script_dir__ / ".." / "data" / "processed" / "aligned_index_prices.csv",
        index_col=0,
        parse_dates=True,
    )
    betas = pd.read_csv(
        __script_dir__ / ".." / "data" / "processed" / "models" / "ordinary_betas_index_only.csv",
        index_col=0,
        parse_dates=True,
    )

    offsets_df = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "close_time_offsets.csv")
    exchange_offsets = dict(zip(offsets_df["exchange_mic"], offsets_df["offset"]))

    exchange_to_stock_currency = defaultdict(list)
    for ordinary_ticker, exchange_mic in ordinary_to_exchange.items():
        stock_currency = ordinary_to_currency.get(ordinary_ticker)
        if stock_currency:
            exchange_to_stock_currency[exchange_mic].append(stock_currency)
    exchange_to_stock_currency = {
        ex: pd.Series(curs).mode().iloc[0] for ex, curs in exchange_to_stock_currency.items() if len(curs) > 0
    }

    fx_minute_cache = {}
    close_times_cache = {}
    fx_daily_by_exchange_currency = {}
    needed_pairs = set()
    for exchange_mic, index_symbol in exchange_to_index.items():
        index_currency = index_to_currency.get(index_symbol)
        stock_currency = exchange_to_stock_currency.get(exchange_mic)
        # Russell returns are already USD. We still need index FX whenever the
        # local index currency is not USD so index returns can be aligned to USD.
        if index_currency and not is_usd_currency(index_currency):
            needed_pairs.add((exchange_mic, index_currency))
        # Ordinary residual path only needs stock FX when stock/index currencies differ.
        if stock_currency and index_currency and stock_currency != index_currency:
            if not is_usd_currency(stock_currency):
                needed_pairs.add((exchange_mic, stock_currency))
    for exchange_mic, currency in sorted(needed_pairs):
        if currency not in fx_minute_cache:
            fx_minute_cache[currency] = load_fx_minute(currency)
        if exchange_mic not in close_times_cache:
            offset_str = exchange_offsets.get(exchange_mic, "0min")
            close_times_cache[exchange_mic] = compute_exchange_close_times(exchange_mic, offset_str, start_date, end_date)
        fx_daily_by_exchange_currency[(exchange_mic, currency)] = compute_fx_daily_at_close(
            fx_minute_cache[currency], close_times_cache[exchange_mic]
        )

    # Fail fast if a non-Asia exchange in the experiment universe is missing its
    # precomputed Russell close-at-auction input. Silent drops here can make the
    # feature universe look valid while excluding entire exchanges.
    required_non_asia_exchanges = {
        exchange_mic
        for ordinary_ticker, exchange_mic in ordinary_to_exchange.items()
        if (
            ordinary_to_adr.get(ordinary_ticker) in experiment_universe
            and exchange_mic not in ASIA_EXCHANGES
        )
    }
    available_russell_exchanges = set(russell_prices_by_exchange.keys())
    missing_non_asia_inputs = sorted(required_non_asia_exchanges - available_russell_exchanges)
    if missing_non_asia_inputs:
        raise RuntimeError(
            "Missing non-Asia Russell exchange feature inputs: "
            f"{', '.join(missing_non_asia_inputs)}. "
            "Run upstream close_at_exchange_auction_adjusted generation first."
        )

    exchange_to_tickers = defaultdict(list)
    for ordinary_ticker, exchange_mic in ordinary_to_exchange.items():
        adr_ticker = ordinary_to_adr.get(ordinary_ticker)
        index_symbol = ordinary_to_index.get(ordinary_ticker)
        if adr_ticker is None or index_symbol is None:
            continue
        if adr_ticker not in experiment_universe:
            continue
        if exchange_mic not in russell_prices_by_exchange and exchange_mic not in ASIA_EXCHANGES:
            continue
        exchange_to_tickers[exchange_mic].append((ordinary_ticker, adr_ticker, index_symbol))

    # NY-close target inputs (defaults to Asia ADRs, overrideable via CLI)
    futures_symbols = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "futures_symbols.csv")
    futures_symbols["bloomberg_symbol"] = futures_symbols["bloomberg_symbol"].astype(str).str.strip()
    adr_info["index_future_bbg"] = adr_info["index_future_bbg"].astype(str).str.strip()
    adr_to_future = (
        adr_info[["adr_ticker", "index_future_bbg"]]
        .merge(futures_symbols[["bloomberg_symbol", "first_rate_symbol"]], left_on="index_future_bbg", right_on="bloomberg_symbol", how="left")
        .set_index("adr_ticker")["first_rate_symbol"]
        .to_dict()
    )
    futures_dir = __script_dir__ / ".." / "data" / "processed" / "futures" / "converted_minute_bars"
    russell_ohlcv_dir = __script_dir__ / ".." / "data" / "raw" / "russell1000" / "ohlcv-1m"

    explicit_ny_close_tickers = None
    if args.ny_close_target_tickers:
        explicit_ny_close_tickers = {
            t.strip() for t in args.ny_close_target_tickers.split(",") if t.strip()
        }
    elif args.ny_close_target_tickers_file:
        explicit_ny_close_tickers = load_ticker_set_from_csv(
            args.ny_close_target_tickers_file,
            column=args.ny_close_target_tickers_column,
        )

    ny_close_target_adrs = resolve_ny_close_target_adrs(
        exchange_to_tickers,
        explicit_tickers=explicit_ny_close_tickers,
    )
    expected_adrs = {
        adr_ticker
        for ticker_list in exchange_to_tickers.values()
        for _, adr_ticker, _ in ticker_list
    }
    unknown_explicit = []
    if explicit_ny_close_tickers is not None:
        unknown_explicit = sorted(set(explicit_ny_close_tickers) - expected_adrs)
    if explicit_ny_close_tickers is None:
        print(
            f"NY-close target ADR mode: default Asia set ({len(ny_close_target_adrs)} tickers).",
            flush=True,
        )
    else:
        print(
            f"NY-close target ADR mode: explicit override ({len(ny_close_target_adrs)} tickers).",
            flush=True,
        )
        if unknown_explicit:
            print(
                f"WARNING: {len(unknown_explicit)} explicit NY-close tickers not in feature universe "
                f"(first 20): {', '.join(unknown_explicit[:20])}",
                flush=True,
            )

    exchange_to_future_symbol = {}
    for ex, ticker_list in exchange_to_tickers.items():
        for _, adr_ticker, _ in ticker_list:
            if adr_ticker not in ny_close_target_adrs:
                continue
            s = adr_to_future.get(adr_ticker)
            if isinstance(s, str) and s:
                exchange_to_future_symbol[ex] = s
                break

    ny_close_target_exchanges = {
        ex for ex, ticker_list in exchange_to_tickers.items()
        if any(adr in ny_close_target_adrs for _, adr, _ in ticker_list)
    }
    canonical_tickers = []
    if russell_prices_by_exchange:
        # Use the full Russell feature universe across exchanges.
        # Taking only the first file's columns can shrink Asia features
        # to a small subset (e.g., ~8 tickers) depending on dict iteration.
        canonical_tickers = sorted(
            {
                col
                for df in russell_prices_by_exchange.values()
                for col in df.columns
            }
        )
    ny_close_to_close_returns = pd.DataFrame()
    if ny_close_target_exchanges:
        ny_close_to_close_returns = _daily_us_close_close_returns(
            russell_ohlcv_dir, canonical_tickers, start_date, end_date
        )

    ny_index_returns = {}
    for ex, sym in exchange_to_future_symbol.items():
        ny_index_returns[ex] = _daily_futures_ny_close_to_close_return(
            futures_dir=futures_dir,
            symbol=sym,
            start_date=start_date,
            end_date=end_date,
        )
    missing_ny_close_future_map = sorted(ny_close_target_exchanges - set(exchange_to_future_symbol.keys()))
    if missing_ny_close_future_map:
        raise RuntimeError(
            "Missing index-future mapping for NY-close exchanges: "
            f"{', '.join(missing_ny_close_future_map)}. "
            "Check adr_info.csv index_future_bbg and futures_symbols.csv first_rate_symbol."
        )

    adr_close = pd.read_csv(
        # Use adjusted daily ADR close for NY-close target residuals to avoid
        # split/corporate-action artifacts contaminating target returns.
        __script_dir__ / ".." / "data" / "raw" / "adrs" / "adr_PX_LAST_adjust_all.csv",
        index_col=0,
        parse_dates=True,
    )
    adr_close.columns = adr_close.columns.str.replace(" US Equity", "", regex=False)
    russell_residuals_local_cache = {}
    russell_residuals_ny_close_cache = {}
    for exchange_mic, ticker_list in exchange_to_tickers.items():
        # Local close-to-close Russell residuals (used by default non-NY-close targets).
        if exchange_mic in russell_prices_by_exchange:
            russell_prices = russell_prices_by_exchange[exchange_mic]
            russell_prices = russell_prices.loc[(russell_prices.index >= start_date) & (russell_prices.index <= end_date)]
            if not russell_prices.empty:
                # Use the best-covered ticker index series for this exchange instead
                # of blindly taking the first ticker, which can silently drop the
                # whole exchange when that ticker's index column is missing/sparse.
                candidate_rep_tickers = [ot for ot, _, _ in ticker_list if ot in aligned_index_prices.columns]
                if candidate_rep_tickers:
                    rep_ticker = max(
                        candidate_rep_tickers,
                        key=lambda t: aligned_index_prices[t].notna().sum(),
                    )
                    index_px = aligned_index_prices[[rep_ticker]].dropna()
                    index_px = index_px.loc[(index_px.index >= start_date) & (index_px.index <= end_date)]

                    russell_returns = compute_aligned_returns(russell_prices)
                    index_returns = compute_aligned_returns(index_px, dates=russell_returns.index)[rep_ticker]

                    index_currency = index_to_currency.get(exchange_to_index.get(exchange_mic))
                    if index_currency and not is_usd_currency(index_currency):
                        index_fx = fx_daily_by_exchange_currency.get((exchange_mic, index_currency))
                        if index_fx is not None:
                            index_returns = convert_returns_to_usd(index_returns, index_fx)

                    russell_residuals_local_cache[exchange_mic] = residualize_returns(russell_returns, index_returns, window=60)

        # NY close-to-close Russell residuals for configured NY-close target tickers.
        if exchange_mic in ny_close_target_exchanges:
            idx_ret_ny = ny_index_returns.get(exchange_mic)
            if idx_ret_ny is not None and not idx_ret_ny.empty and not ny_close_to_close_returns.empty:
                rr = ny_close_to_close_returns.loc[
                    (ny_close_to_close_returns.index >= start_date) &
                    (ny_close_to_close_returns.index <= end_date)
                ]
                idx_ret_ny = idx_ret_ny.loc[(idx_ret_ny.index >= rr.index.min()) & (idx_ret_ny.index <= rr.index.max())]
                if not rr.empty and not idx_ret_ny.empty:
                    russell_residuals_ny_close_cache[exchange_mic] = residualize_returns(rr, idx_ret_ny, window=60)

    success_count = 0
    skip_count = 0
    low_feature_count = 0
    total_selected_features = 0
    failures = []
    written_adrs = set()
    for exchange_mic, ticker_list in exchange_to_tickers.items():
        for ordinary_ticker, adr_ticker, _ in ticker_list:
            try:
                use_ny_close_target = adr_ticker in ny_close_target_adrs
                if use_ny_close_target:
                    russell_residuals = russell_residuals_ny_close_cache.get(exchange_mic)
                    if russell_residuals is None or russell_residuals.empty:
                        skip_count += 1
                        continue
                    if adr_ticker not in adr_close.columns:
                        skip_count += 1
                        continue
                    adr_px = adr_close[[adr_ticker]].dropna()
                    adr_px = adr_px.loc[(adr_px.index >= start_date) & (adr_px.index <= end_date)]
                    if adr_px.empty:
                        skip_count += 1
                        continue
                    adr_ret = compute_aligned_returns(adr_px)[adr_ticker]
                    idx_ret = ny_index_returns.get(exchange_mic, pd.Series(dtype=float))
                    if adr_ticker not in betas.columns or idx_ret.empty:
                        skip_count += 1
                        continue
                    beta_series = betas[adr_ticker].dropna()
                    common = adr_ret.index.intersection(idx_ret.index).intersection(beta_series.index)
                    if len(common) == 0:
                        skip_count += 1
                        continue
                    ordinary_residuals = adr_ret.loc[common] - beta_series.loc[common] * idx_ret.loc[common]
                else:
                    russell_residuals = russell_residuals_local_cache.get(exchange_mic)
                    if russell_residuals is None or russell_residuals.empty:
                        skip_count += 1
                        continue
                    ordinary_px = ordinary_prices[[ordinary_ticker]].dropna()
                    ordinary_px = ordinary_px.loc[(ordinary_px.index >= start_date) & (ordinary_px.index <= end_date)]
                    if ordinary_px.empty:
                        skip_count += 1
                        continue
                    ordinary_returns = compute_aligned_returns(ordinary_px)[ordinary_ticker]

                    if ordinary_ticker not in aligned_index_prices.columns:
                        skip_count += 1
                        continue
                    idx_px = aligned_index_prices[[ordinary_ticker]].dropna()
                    idx_px = idx_px.loc[(idx_px.index >= start_date) & (idx_px.index <= end_date)]
                    index_returns = compute_aligned_returns(idx_px, dates=ordinary_returns.index)[ordinary_ticker]

                    stock_currency = ordinary_to_currency.get(ordinary_ticker)
                    index_currency = index_to_currency.get(ordinary_to_index.get(ordinary_ticker))
                    if stock_currency and index_currency and stock_currency != index_currency:
                        stock_fx = fx_daily_by_exchange_currency.get((exchange_mic, stock_currency))
                        index_fx = fx_daily_by_exchange_currency.get((exchange_mic, index_currency))
                        if not is_usd_currency(stock_currency) and stock_fx is not None:
                            ordinary_returns = convert_returns_to_usd(ordinary_returns, stock_fx)
                        if not is_usd_currency(index_currency) and index_fx is not None:
                            index_returns = convert_returns_to_usd(index_returns, index_fx)

                    ordinary_residuals = get_existing_beta_residuals(
                        ordinary_ticker, adr_ticker, ordinary_returns, index_returns, betas
                    )

                common_dates = ordinary_residuals.index.intersection(russell_residuals.index)
                if len(common_dates) == 0:
                    skip_count += 1
                    continue

                features = build_feature_matrix(ordinary_residuals, russell_residuals)
                if features is None:
                    low_feature_count += 1
                    skip_count += 1
                    continue
                total_selected_features += len([c for c in features.columns if c.startswith("russell_")])
                features.to_parquet(output_dir / f"{adr_ticker}.parquet")
                success_count += 1
                written_adrs.add(adr_ticker)
            except Exception as e:
                skip_count += 1
                failures.append(
                    {
                        "exchange": exchange_mic,
                        "ordinary_ticker": ordinary_ticker,
                        "adr_ticker": adr_ticker,
                        "error_type": e.__class__.__name__,
                        "error": str(e),
                    }
                )
                print(
                    f"ERROR: feature build failed for {adr_ticker} ({exchange_mic}) - "
                    f"{e.__class__.__name__}: {e}",
                    flush=True,
                )
                print(traceback.format_exc(limit=2), flush=True)

    avg_features = (total_selected_features / success_count) if success_count > 0 else 0.0
    print(
        "Extended feature preparation complete: "
        f"success={success_count}, skipped={skip_count}, "
        f"low_feature_count={low_feature_count}, avg_selected_features={avg_features:.1f}"
    )
    if failures:
        err_path = output_dir.parent / f"{output_dir.name}_errors.csv"
        pd.DataFrame(failures).to_csv(err_path, index=False)
        print(f"Wrote feature exception audit: {err_path} ({len(failures)} rows)")
    validate_feature_outputs(
        output_dir=output_dir,
        expected_adrs=expected_adrs,
        min_feature_count=FEATURE_MIN_COUNT,
        min_success_ratio=args.min_success_ratio,
        run_written_adrs=written_adrs,
    )


if __name__ == "__main__":
    main()
