"""
Prepare extended-history Russell residual features for model experiments.

Adds an AU/JP-specific path:
- target is ADR return from ordinary-theoretical close (USD) to ADR close (USD),
  residualized by existing index-only betas and futures returns
- Russell features use same-day US regular-session open->close returns
"""

from collections import defaultdict
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from utils_lasso_residuals import (
    load_ordinary_exchange_mapping,
    load_index_mapping,
    compute_aligned_returns,
    residualize_returns,
    get_existing_beta_residuals,
    fill_missing_values,
    INDEX_TO_FX_CURRENCY,
    load_fx_minute,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)

__script_dir__ = Path(__file__).parent.absolute()
ASIA_EXCHANGES = {"XTKS", "XASX"}


def load_experiment_universe():
    adr_info = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "adr_info.csv")
    adr_info["adr_ticker"] = adr_info["adr"].str.replace(" US Equity", "", regex=False)
    return set(adr_info["adr_ticker"].dropna().unique().tolist())


def _daily_us_open_close_returns(russell_ohlcv_dir, tickers, start_date, end_date):
    data = {}
    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    for t in tickers:
        p = russell_ohlcv_dir / f"ticker={t}" / "data.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["Close", "date"])
            df = df[(df["date"] >= start_s) & (df["date"] <= end_s)]
            if df.empty:
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize("America/New_York")
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
    params = load_params()
    start_date = params["start_date"]
    end_date = params["end_date"]
    experiment_universe = load_experiment_universe()

    ordinary_to_exchange, ordinary_to_adr = load_ordinary_exchange_mapping(include_asia=True)
    ordinary_to_index, exchange_to_index = load_index_mapping(include_asia=True)
    adr_info = pd.read_csv(__script_dir__ / ".." / "data" / "raw" / "adr_info.csv")
    adr_info["currency"] = adr_info["currency"].replace({"GBp": "GBP"})
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

    # Asia inputs
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
    asia_exchange_to_symbol = {}
    for ex in ASIA_EXCHANGES:
        for _, adr_ticker, _ in exchange_to_tickers.get(ex, []):
            s = adr_to_future.get(adr_ticker)
            if isinstance(s, str) and s:
                asia_exchange_to_symbol[ex] = s
                break
    canonical_tickers = []
    if russell_prices_by_exchange:
        canonical_tickers = list(next(iter(russell_prices_by_exchange.values())).columns)
    us_open_close_returns = pd.DataFrame()
    if any(ex in exchange_to_tickers for ex in ASIA_EXCHANGES):
        us_open_close_returns = _daily_us_open_close_returns(russell_ohlcv_dir, canonical_tickers, start_date, end_date)

    asia_index_returns = {}
    for ex, sym in asia_exchange_to_symbol.items():
        asia_index_returns[ex] = _daily_futures_domestic_to_ny_return(
            futures_dir=futures_dir,
            symbol=sym,
            exchange_mic=ex,
            start_date=start_date,
            end_date=end_date,
            exchange_offsets=exchange_offsets,
        )

    adr_close = pd.read_csv(
        __script_dir__ / ".." / "data" / "raw" / "adrs" / "adr_PX_LAST_adjust_none.csv",
        index_col=0,
        parse_dates=True,
    )
    adr_close.columns = adr_close.columns.str.replace(" US Equity", "", regex=False)
    ord_theo_close = pd.read_csv(
        __script_dir__ / ".." / "data" / "processed" / "ordinary" / "ord_close_to_usd_adr_PX_LAST_adjust_none.csv",
        index_col=0,
        parse_dates=True,
    )
    id_to_adr = dict(zip(adr_info["id"], adr_info["adr_ticker"]))
    ord_theo_close = ord_theo_close.rename(columns=id_to_adr)

    russell_residuals_cache = {}
    for exchange_mic, ticker_list in exchange_to_tickers.items():
        if exchange_mic in ASIA_EXCHANGES:
            if us_open_close_returns.empty:
                continue
            idx_ret = asia_index_returns.get(exchange_mic)
            if idx_ret is None or idx_ret.empty:
                continue
            rr = us_open_close_returns.loc[(us_open_close_returns.index >= start_date) & (us_open_close_returns.index <= end_date)]
            idx_ret = idx_ret.loc[(idx_ret.index >= rr.index.min()) & (idx_ret.index <= rr.index.max())]
            if rr.empty or idx_ret.empty:
                continue
            russell_residuals_cache[exchange_mic] = residualize_returns(rr, idx_ret, window=60)
            continue

        russell_prices = russell_prices_by_exchange[exchange_mic]
        russell_prices = russell_prices.loc[(russell_prices.index >= start_date) & (russell_prices.index <= end_date)]
        if russell_prices.empty:
            continue

        rep_ticker = ticker_list[0][0]
        if rep_ticker not in aligned_index_prices.columns:
            continue
        index_px = aligned_index_prices[[rep_ticker]].dropna()
        index_px = index_px.loc[(index_px.index >= start_date) & (index_px.index <= end_date)]

        russell_returns = compute_aligned_returns(russell_prices)
        index_returns = compute_aligned_returns(index_px, dates=russell_returns.index)[rep_ticker]

        stock_currency = exchange_to_stock_currency.get(exchange_mic)
        index_currency = INDEX_TO_FX_CURRENCY.get(exchange_to_index.get(exchange_mic))
        if stock_currency and index_currency and stock_currency != index_currency:
            stock_fx = fx_daily_by_exchange_currency.get((exchange_mic, stock_currency))
            index_fx = fx_daily_by_exchange_currency.get((exchange_mic, index_currency))
            if stock_fx is not None:
                russell_returns = convert_returns_to_usd(russell_returns, stock_fx)
            if index_fx is not None:
                index_returns = convert_returns_to_usd(index_returns, index_fx)

        russell_residuals_cache[exchange_mic] = residualize_returns(russell_returns, index_returns, window=60)

    output_dir = __script_dir__ / ".." / "data" / "processed" / "models" / "with_us_stocks" / "features_extended"
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    for exchange_mic, ticker_list in exchange_to_tickers.items():
        russell_residuals = russell_residuals_cache.get(exchange_mic)
        if russell_residuals is None or russell_residuals.empty:
            skip_count += len(ticker_list)
            continue

        for ordinary_ticker, adr_ticker, _ in ticker_list:
            try:
                if exchange_mic in ASIA_EXCHANGES:
                    if adr_ticker not in adr_close.columns or adr_ticker not in ord_theo_close.columns:
                        skip_count += 1
                        continue
                    adr_ret = (adr_close[adr_ticker] / ord_theo_close[adr_ticker] - 1.0).dropna()
                    idx_ret = asia_index_returns.get(exchange_mic, pd.Series(dtype=float))
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
                    index_currency = INDEX_TO_FX_CURRENCY.get(ordinary_to_index.get(ordinary_ticker))
                    if stock_currency and index_currency and stock_currency != index_currency:
                        stock_fx = fx_daily_by_exchange_currency.get((exchange_mic, stock_currency))
                        index_fx = fx_daily_by_exchange_currency.get((exchange_mic, index_currency))
                        if stock_fx is not None:
                            ordinary_returns = convert_returns_to_usd(ordinary_returns, stock_fx)
                        if index_fx is not None:
                            index_returns = convert_returns_to_usd(index_returns, index_fx)

                    ordinary_residuals = get_existing_beta_residuals(
                        ordinary_ticker, adr_ticker, ordinary_returns, index_returns, betas
                    )

                common_dates = ordinary_residuals.index.intersection(russell_residuals.index)
                if len(common_dates) == 0:
                    skip_count += 1
                    continue

                features = pd.DataFrame(index=common_dates)
                features["ordinary_residual"] = ordinary_residuals.loc[common_dates]
                r_aligned = russell_residuals.loc[common_dates]
                r_features = r_aligned.copy()
                r_features.columns = [f"russell_{col}" for col in r_features.columns]
                features = pd.concat([features, r_features], axis=1)
                features = fill_missing_values(features, fill_value=0.0)
                features.to_parquet(output_dir / f"{adr_ticker}.parquet")
                success_count += 1
            except Exception:
                skip_count += 1

    print(f"Extended feature preparation complete: success={success_count}, skipped={skip_count}")


if __name__ == "__main__":
    main()
