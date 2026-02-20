"""
Compute rolling 60-day betas for Russell features vs index return.

Standard exchanges use close-at-foreign-auction series.
AU/JP (XASX/XTKS) use same-day US regular-session open->close returns.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from utils_lasso_residuals import (
    load_index_mapping,
    load_ordinary_exchange_mapping,
    INDEX_TO_FX_CURRENCY,
    load_fx_minute,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)

__script_dir__ = Path(__file__).parent.absolute()
ASIA_EXCHANGES = {"XTKS", "XASX"}
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


def _daily_us_open_close_returns(russell_ohlcv_dir, tickers, start_date, end_date):
    data = {}
    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    for t in tickers:
        p = russell_ohlcv_dir / f"ticker={t}" / "data.parquet"
        if not p.exists():
            continue
        try:
            # Use only the Parquet DateTime index for time filtering.
            df = pd.read_parquet(p, columns=["Close"])
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize("America/New_York")
            else:
                df.index = df.index.tz_convert("America/New_York")
            day_str = df.index.strftime("%Y-%m-%d")
            df = df[(day_str >= start_s) & (day_str <= end_s)]
            if df.empty:
                continue
            intraday = df.between_time("09:30", "16:00", inclusive="both")
            if intraday.empty:
                continue
            g = intraday.groupby(intraday.index.strftime("%Y-%m-%d"))["Close"].agg(["first", "last"])
            ret = (g["last"] / g["first"] - 1.0).rename(t)
            ret.index = pd.to_datetime(ret.index)
            data[t] = ret
        except Exception:
            continue
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).sort_index()


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


def main():
    print("=" * 70)
    print("Computing Russell 1000 Rolling Betas vs Index")
    print("=" * 70)

    params = load_params()
    start_date = params["frd_start_date"]
    end_date = params["end_date"]
    print(f"Date range: {start_date} to {end_date}")

    _, exchange_to_index = load_index_mapping(include_asia=True)
    ordinary_to_exchange, _ = load_ordinary_exchange_mapping(include_asia=True)
    adr_info = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "adr_info.csv")
    adr_info["currency"] = adr_info["currency"].replace({"GBp": "GBP"})
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
        index_currency = INDEX_TO_FX_CURRENCY.get(index_symbol)
        stock_currency = exchange_to_stock_currency.get(exchange_mic)
        if stock_currency and index_currency and stock_currency != index_currency:
            needed_pairs.add((exchange_mic, stock_currency))
            needed_pairs.add((exchange_mic, index_currency))
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
        if index_symbol is None:
            continue
        print(f"\n  Processing {exchange_mic} (index: {index_symbol})...")
        russell_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        rep_ticker = exchange_to_rep_ticker.get(exchange_mic)
        if rep_ticker is None:
            continue
        index_px = aligned_index_prices[rep_ticker].dropna()
        russell_returns = russell_prices.pct_change()
        index_ret = index_px.pct_change()

        stock_currency = exchange_to_stock_currency.get(exchange_mic)
        index_currency = INDEX_TO_FX_CURRENCY.get(index_symbol)
        if stock_currency and index_currency and stock_currency != index_currency:
            stock_fx = fx_daily_by_exchange_currency.get((exchange_mic, stock_currency))
            index_fx = fx_daily_by_exchange_currency.get((exchange_mic, index_currency))
            if stock_fx is not None:
                russell_returns = convert_returns_to_usd(russell_returns, stock_fx)
            if index_fx is not None:
                index_ret = convert_returns_to_usd(index_ret, index_fx)

        betas = compute_rolling_betas(russell_returns, index_ret).dropna(how="all")
        out = output_dir / f"{exchange_mic}.parquet"
        betas.to_parquet(out)
        print(f"    Betas shape: {betas.shape} -> {out}")

    # AU/JP synthetic exchanges
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
    us_open_close_returns = _daily_us_open_close_returns(russell_ohlcv_dir, canonical_tickers, start_date, end_date)

    for exchange_mic in sorted(ASIA_EXCHANGES):
        tickers = adr_info[adr_info["exchange"] == exchange_mic]["adr_ticker"].tolist()
        fut_sym = None
        for t in tickers:
            s = adr_to_future.get(t)
            if isinstance(s, str) and s:
                fut_sym = s
                break
        if fut_sym is None or us_open_close_returns.empty:
            continue
        idx_ret = _daily_futures_domestic_to_ny_return(
            futures_dir=futures_dir,
            symbol=fut_sym,
            exchange_mic=exchange_mic,
            start_date=start_date,
            end_date=end_date,
            exchange_offsets=exchange_offsets,
        )
        if idx_ret.empty:
            continue
        rr = us_open_close_returns.copy()
        stock_currency = exchange_to_stock_currency.get(exchange_mic)
        index_currency = INDEX_TO_FX_CURRENCY.get(exchange_to_index.get(exchange_mic))
        if stock_currency and index_currency and stock_currency != index_currency:
            stock_fx = fx_daily_by_exchange_currency.get((exchange_mic, stock_currency))
            index_fx = fx_daily_by_exchange_currency.get((exchange_mic, index_currency))
            if stock_fx is not None:
                rr = convert_returns_to_usd(rr, stock_fx)
            if index_fx is not None:
                idx_ret = convert_returns_to_usd(idx_ret, index_fx)

        betas = compute_rolling_betas(rr, idx_ret).dropna(how="all")
        out = output_dir / f"{exchange_mic}.parquet"
        betas.to_parquet(out)
        print(f"    {exchange_mic} synthetic betas: {betas.shape} -> {out}")

    print("\n" + "=" * 70)
    print("Russell betas computation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
