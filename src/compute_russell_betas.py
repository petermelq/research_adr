"""
Compute rolling 60-day betas for Russell features vs index return.

Standard exchanges use close-at-foreign-auction series.
Asia exchanges (AU/JP/HK/CN/SG) use same-day US regular-session open->close returns.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from utils_lasso_residuals import (
    load_index_mapping,
    load_ordinary_exchange_mapping,
    load_index_currency_mapping,
    load_fx_minute,
    is_usd_currency,
    normalize_currency,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)

__script_dir__ = Path(__file__).parent.absolute()
ASIA_EXCHANGES = {"XTKS", "XASX", "XHKG", "XSES", "XSHG", "XSHE"}
WINDOW = 60
MIN_OBS = 20


def compute_rolling_betas(russell_returns, index_returns, window=WINDOW, min_obs=MIN_OBS):
    common_dates = russell_returns.index.intersection(index_returns.index)
    R = russell_returns.loc[common_dates]
    I = index_returns.loc[common_dates]
    RI = R.multiply(I, axis=0)
    roll_mean_RI = RI.rolling(window, min_periods=min_obs).mean()
    roll_mean_R = R.rolling(window, min_periods=min_obs).mean()
    roll_mean_I = I.rolling(window, min_periods=min_obs).mean()
    roll_cov = roll_mean_RI.subtract(roll_mean_R.multiply(roll_mean_I, axis=0))
    idx_var = I.rolling(window, min_periods=min_obs).var(ddof=0)
    betas = roll_cov.divide(idx_var, axis=0)
    betas[idx_var == 0] = 0
    return betas


def _nyse_date_sets(start_date, end_date):
    nyse_sched = mcal.get_calendar("NYSE").schedule(start_date=start_date, end_date=end_date)
    close_times_et = nyse_sched["market_close"].dt.tz_convert("America/New_York")
    close_tod = close_times_et.dt.time
    normal_close_tod = close_tod.mode().iloc[0]
    is_normal = close_tod == normal_close_tod
    all_dates = pd.DatetimeIndex(nyse_sched.index).tz_localize(None)
    normal_dates = pd.DatetimeIndex(nyse_sched.index[is_normal]).tz_localize(None)
    return all_dates, normal_dates


def _extract_russell_ny_close_for_ticker(parquet_path, normal_dates_set):
    try:
        df = pd.read_parquet(parquet_path, columns=["Close"])
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("America/New_York")
        else:
            idx = idx.tz_convert("America/New_York")
        is_ny_close = (idx.hour == 16) & (idx.minute == 0)
        if not is_ny_close.any():
            return None
        close_vals = df["Close"].to_numpy(dtype=np.float32)[is_ny_close]
        close_df = pd.DataFrame({"Close": close_vals}, index=idx[is_ny_close])
        close_df["date"] = pd.to_datetime(close_df.index.strftime("%Y-%m-%d"))
        close_df = close_df[close_df["date"].isin(normal_dates_set)]
        if close_df.empty:
            return None
        s = close_df.groupby("date")["Close"].last().astype(np.float32)
        return s
    except Exception:
        return None


def _daily_russell_ny_close_returns_parallel(russell_ohlcv_dir, tickers, normal_dates):
    normal_dates_set = set(normal_dates)
    tasks = []
    for t in tickers:
        p = russell_ohlcv_dir / f"ticker={t}" / "data.parquet"
        if p.exists():
            tasks.append((t, p))
    if not tasks:
        return pd.DataFrame()

    data = {}
    max_workers = min(12, max(1, os.cpu_count() or 1))
    print(f"  Asia: extracting NY 16:00 closes in parallel for {len(tasks)} tickers ({max_workers} workers)")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_extract_russell_ny_close_for_ticker, p, normal_dates_set): t for t, p in tasks}
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            s = fut.result()
            if s is not None and not s.empty:
                data[t] = s
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"    processed {done}/{len(tasks)} tickers")
    if not data:
        return pd.DataFrame()
    close_px = pd.DataFrame(data).sort_index()
    return close_px.pct_change(fill_method=None)


def _daily_futures_ny_close_returns(futures_dir, symbol, start_date, normal_dates):
    start_ts = pd.Timestamp(start_date, tz="America/New_York")
    fut = pd.read_parquet(
        futures_dir,
        filters=[("symbol", "==", symbol), ("timestamp", ">=", start_ts)],
        columns=["timestamp", "symbol", "close"],
    ).sort_values("timestamp")
    if fut.empty:
        return pd.Series(dtype=float)
    idx = fut["timestamp"]
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize("America/New_York")
    else:
        idx = idx.dt.tz_convert("America/New_York")
    fut = fut.set_index(idx)
    # Use last print at or before NY 16:00 (not exact 16:00), since some contracts
    # can skip that exact minute.
    fut_df = pd.DataFrame({"close": fut["close"].to_numpy(dtype=np.float64)}, index=fut.index)
    fut_df["date"] = pd.to_datetime(fut_df.index.strftime("%Y-%m-%d"))
    fut_df["tod"] = fut_df.index.time
    cutoff = pd.Timestamp("16:00").time()
    fut_df = fut_df[fut_df["tod"] <= cutoff]
    if fut_df.empty:
        return pd.Series(dtype=float)
    close_px = fut_df.groupby("date")["close"].last()
    close_px = close_px[close_px.index.isin(normal_dates)]
    ret = close_px.pct_change(fill_method=None)
    return ret.sort_index()


def main():
    print("=" * 70)
    print("Computing Russell 1000 Rolling Betas vs Index")
    print("=" * 70)

    params = load_params()
    start_date = params["frd_start_date"]
    end_date = params["end_date"]
    beta_lookback = int(params.get("russell_betas", {}).get("lookback_days", WINDOW))
    print(f"Date range: {start_date} to {end_date}")
    print(f"Russell beta lookback window: {beta_lookback} days")

    _, exchange_to_index = load_index_mapping(include_asia=True)
    index_to_currency = load_index_currency_mapping()
    ordinary_to_exchange, _ = load_ordinary_exchange_mapping(include_asia=True)
    adr_info = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "adr_info.csv")
    adr_info["currency"] = adr_info["currency"].map(normalize_currency)
    adr_info["adr_ticker"] = adr_info["adr"].str.replace(" US Equity", "", regex=False)
    ordinary_to_currency = dict(zip(adr_info["id"], adr_info["currency"]))
    print(f"Exchange-to-index mapping: {exchange_to_index}")

    aligned_index_prices = pd.read_csv(
        __script_dir__ / ".." / "data" / "processed" / "aligned_index_prices.csv",
        index_col=0,
        parse_dates=True,
    )
    exchange_to_rep_ticker = {}
    for ordinary_ticker, exchange_mic in ordinary_to_exchange.items():
        if ordinary_ticker in aligned_index_prices.columns and exchange_mic not in exchange_to_rep_ticker:
            exchange_to_rep_ticker[exchange_mic] = ordinary_ticker

    input_dir = __script_dir__ / ".." / "data" / "processed" / "russell1000" / "close_at_exchange_auction_adjusted"
    output_dir = __script_dir__ / ".." / "data" / "processed" / "russell1000" / "russell_betas"
    output_dir.mkdir(parents=True, exist_ok=True)

    offsets_df = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "close_time_offsets.csv")
    exchange_offsets = dict(zip(offsets_df["exchange_mic"], offsets_df["offset"]))
    nyse_all_dates, nyse_normal_dates = _nyse_date_sets(start_date, end_date)
    print(f"NYSE normal-close dates: {len(nyse_normal_dates)} / total NYSE dates: {len(nyse_all_dates)}")

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
        if stock_currency and index_currency and stock_currency != index_currency:
            if not is_usd_currency(stock_currency):
                needed_pairs.add((exchange_mic, stock_currency))
            if not is_usd_currency(index_currency):
                needed_pairs.add((exchange_mic, index_currency))
    required_currencies = sorted({currency for _, currency in needed_pairs})
    missing_currency_files = []
    for currency in required_currencies:
        fx_file = __script_dir__ / ".." / "data" / "raw" / "currencies" / "minute_bars" / f"{currency}USD_full_1min.txt"
        if not fx_file.exists():
            missing_currency_files.append(str(fx_file))
    if missing_currency_files:
        missing_str = "\n".join(missing_currency_files)
        raise FileNotFoundError(
            "Missing required FX minute-bar files for compute_russell_betas:\n"
            f"{missing_str}\n"
            "Run upstream FX download/inversion stages first."
        )

    for exchange_mic, currency in sorted(needed_pairs):
        if currency not in fx_minute_cache:
            fx_minute_cache[currency] = load_fx_minute(currency)
        if exchange_mic not in close_times_cache:
            offset_str = exchange_offsets.get(exchange_mic, "0min")
            close_times_cache[exchange_mic] = compute_exchange_close_times(exchange_mic, offset_str, start_date, end_date)
        fx_daily_by_exchange_currency[(exchange_mic, currency)] = compute_fx_daily_at_close(
            fx_minute_cache[currency], close_times_cache[exchange_mic]
        )

    # Standard exchanges
    for csv_file in sorted(input_dir.glob("*.csv")):
        exchange_mic = csv_file.stem
        index_symbol = exchange_to_index.get(exchange_mic)
        if index_symbol is None or pd.isna(index_symbol):
            print(f"\n  Skipping {exchange_mic}: no mapped index symbol")
            continue
        print(f"\n  Processing {exchange_mic} (index: {index_symbol})...")
        russell_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        rep_ticker = exchange_to_rep_ticker.get(exchange_mic)
        if rep_ticker is None:
            continue
        index_px = aligned_index_prices[rep_ticker].dropna()
        russell_returns = russell_prices.pct_change(fill_method=None)
        index_ret = index_px.pct_change(fill_method=None)

        stock_currency = exchange_to_stock_currency.get(exchange_mic)
        index_currency = index_to_currency.get(index_symbol)
        if stock_currency and index_currency and stock_currency != index_currency:
            stock_fx = fx_daily_by_exchange_currency.get((exchange_mic, stock_currency))
            index_fx = fx_daily_by_exchange_currency.get((exchange_mic, index_currency))
            if not is_usd_currency(stock_currency) and stock_fx is not None:
                russell_returns = convert_returns_to_usd(russell_returns, stock_fx)
            if not is_usd_currency(index_currency) and index_fx is not None:
                index_ret = convert_returns_to_usd(index_ret, index_fx)

        betas = compute_rolling_betas(russell_returns, index_ret, window=beta_lookback).dropna(how="all")
        out = output_dir / f"{exchange_mic}.parquet"
        betas.to_parquet(out)
        print(f"    Betas shape: {betas.shape} -> {out}")

    # Asia synthetic exchanges
    futures_symbols = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "futures_symbols.csv")
    futures_symbols["bloomberg_symbol"] = futures_symbols["bloomberg_symbol"].astype(str).str.strip()
    adr_info["index_future_bbg"] = adr_info["index_future_bbg"].astype(str).str.strip()
    adr_to_future = (
        adr_info[["adr_ticker", "index_future_bbg"]]
        .merge(
            futures_symbols[["bloomberg_symbol", "first_rate_symbol"]],
            left_on="index_future_bbg",
            right_on="bloomberg_symbol",
            how="left",
        )
        .set_index("adr_ticker")["first_rate_symbol"]
        .to_dict()
    )
    futures_dir = __script_dir__ / ".." / "data" / "processed" / "futures" / "converted_minute_bars"
    russell_ohlcv_dir = __script_dir__ / ".." / "data" / "raw" / "russell1000" / "ohlcv-1m"
    if list(input_dir.glob("*.csv")):
        canonical_tickers = list(pd.read_csv(sorted(input_dir.glob("*.csv"))[0], index_col=0, nrows=1).columns)
    else:
        canonical_tickers = []
    us_close_to_close_returns = _daily_russell_ny_close_returns_parallel(
        russell_ohlcv_dir, canonical_tickers, nyse_normal_dates
    )

    for exchange_mic in sorted(ASIA_EXCHANGES):
        tickers = adr_info[adr_info["exchange"] == exchange_mic]["adr_ticker"].tolist()
        fut_sym = None
        for t in tickers:
            s = adr_to_future.get(t)
            if isinstance(s, str) and s:
                fut_sym = s
                break
        if fut_sym is None or us_close_to_close_returns.empty:
            continue
        idx_ret = _daily_futures_ny_close_returns(
            futures_dir=futures_dir,
            symbol=fut_sym,
            start_date=start_date,
            normal_dates=nyse_normal_dates,
        )
        if idx_ret.empty:
            continue
        rr = us_close_to_close_returns
        stock_currency = exchange_to_stock_currency.get(exchange_mic)
        index_currency = index_to_currency.get(exchange_to_index.get(exchange_mic))
        if stock_currency and index_currency and stock_currency != index_currency:
            stock_fx = fx_daily_by_exchange_currency.get((exchange_mic, stock_currency))
            index_fx = fx_daily_by_exchange_currency.get((exchange_mic, index_currency))
            if not is_usd_currency(stock_currency) and stock_fx is not None:
                rr = convert_returns_to_usd(rr, stock_fx)
            if not is_usd_currency(index_currency) and index_fx is not None:
                idx_ret = convert_returns_to_usd(idx_ret, index_fx)

        betas = compute_rolling_betas(rr, idx_ret, window=beta_lookback).dropna(how="all")
        # Reinsert early-close NYSE dates by forward-filling from prior normal-close beta.
        betas = betas.reindex(nyse_all_dates).ffill()
        out = output_dir / f"{exchange_mic}.parquet"
        betas.to_parquet(out)
        print(f"    {exchange_mic} NY-close synthetic betas: {betas.shape} -> {out}")

    print("\n" + "=" * 70)
    print("Russell betas computation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
