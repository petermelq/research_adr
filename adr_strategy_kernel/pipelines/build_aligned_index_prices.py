from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from adr_strategy_kernel.common import ensure_parent_dir, load_adr_info, resolve_repo_path


def parse_minutes_since_midnight(value: str) -> int:
    parts = str(value).split(":")
    return int(parts[0]) * 60 + int(parts[1])


def identify_misaligned_stocks(
    tickers: list[str],
    close_times_df: pd.DataFrame,
    stock_to_index: dict[str, str | None],
) -> set[str]:
    misaligned_stocks: set[str] = set()
    for ticker in tickers:
        idx_symbol = stock_to_index.get(ticker)
        if idx_symbol is None:
            continue
        idx_ticker = f"{idx_symbol} Index"
        if ticker not in close_times_df.index or idx_ticker not in close_times_df.index:
            continue
        stock_min = parse_minutes_since_midnight(close_times_df.loc[ticker, "BLOOMBERG_CLOSE_TIME"])
        index_min = parse_minutes_since_midnight(close_times_df.loc[idx_ticker, "BLOOMBERG_CLOSE_TIME"])
        if abs(stock_min - index_min) > 10:
            misaligned_stocks.add(ticker)
    return misaligned_stocks


def sample_futures_at_close(
    futures_path: str | Path,
    close_times: pd.Series,
) -> pd.Series:
    futures_df = pd.read_csv(
        futures_path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume"],
    )
    futures_df["timestamp"] = pd.to_datetime(
        futures_df["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
    ).dt.tz_localize("America/New_York")
    futures_minute = futures_df.set_index("timestamp")["close"]

    futures_idx_int = futures_minute.index.values.astype("int64")
    close_times_int = close_times.values.astype("int64")
    indices = np.searchsorted(futures_idx_int, close_times_int, side="right") - 1
    valid = indices >= 0

    futures_at_close = pd.Series(index=close_times.index, dtype=float)
    futures_at_close.loc[close_times.index[valid]] = futures_minute.values[indices[valid]]
    futures_at_close = futures_at_close.dropna()
    futures_at_close.index = pd.to_datetime(futures_at_close.index.date)
    return futures_at_close


def build_aligned_index_prices(
    adr_info_path: str | Path | None = None,
    futures_symbols_path: str | Path | None = None,
    index_prices_path: str | Path | None = None,
    bloomberg_close_times_path: str | Path | None = None,
    close_offsets_path: str | Path | None = None,
    futures_minute_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    adr_info = load_adr_info(adr_info_path=adr_info_path, repo_root=repo_root)
    adr_info = adr_info.dropna(subset=["adr"])
    adr_info = adr_info[~adr_info["id"].str.contains(" US Equity")]
    tickers = adr_info["id"].tolist()

    futures_symbols_file = Path(futures_symbols_path) if futures_symbols_path is not None else resolve_repo_path(
        "data", "raw", "futures_symbols.csv", repo_root=repo_root
    )
    futures_symbols = pd.read_csv(futures_symbols_file)
    futures_symbols["bloomberg_symbol"] = futures_symbols["bloomberg_symbol"].astype(str).str.strip()
    futures_symbols["first_rate_symbol"] = futures_symbols["first_rate_symbol"].astype(str).str.strip()
    futures_to_index = futures_symbols.set_index("bloomberg_symbol")["index"].to_dict()
    bbg_to_frd = (
        futures_symbols[["bloomberg_symbol", "first_rate_symbol"]]
        .dropna(subset=["first_rate_symbol"])
        .set_index("bloomberg_symbol")["first_rate_symbol"]
        .to_dict()
    )

    stock_to_index_future = adr_info.set_index("id")["index_future_bbg"].to_dict()
    stock_to_index = {
        stock: futures_to_index.get(index_future)
        for stock, index_future in stock_to_index_future.items()
    }
    stock_to_exchange = adr_info.set_index("id")["exchange"].to_dict()

    index_prices_file = Path(index_prices_path) if index_prices_path is not None else resolve_repo_path(
        "data", "raw", "indices", "indices_PX_LAST.csv", repo_root=repo_root
    )
    index_data = pd.read_csv(index_prices_file, index_col=0, parse_dates=True)

    bloomberg_close_times_file = (
        Path(bloomberg_close_times_path)
        if bloomberg_close_times_path is not None
        else resolve_repo_path("data", "raw", "bloomberg_close_times.csv", repo_root=repo_root)
    )
    close_times_df = pd.read_csv(bloomberg_close_times_file, index_col=0)

    close_offsets_file = Path(close_offsets_path) if close_offsets_path is not None else resolve_repo_path(
        "data", "raw", "close_time_offsets.csv", repo_root=repo_root
    )
    offsets_df = pd.read_csv(close_offsets_file)
    offsets = dict(zip(offsets_df["exchange_mic"], offsets_df["offset"]))

    misaligned_stocks = identify_misaligned_stocks(tickers, close_times_df, stock_to_index)

    data_start = index_data.index.min().strftime("%Y-%m-%d")
    data_end = index_data.index.max().strftime("%Y-%m-%d")
    futures_dir = Path(futures_minute_dir) if futures_minute_dir is not None else resolve_repo_path(
        "data", "raw", "futures", "minute_bars", repo_root=repo_root
    )

    exchange_futures_at_close: dict[str, pd.Series] = {}
    for ticker in misaligned_stocks:
        exchange = stock_to_exchange[ticker]
        if exchange in exchange_futures_at_close:
            continue

        index_future = stock_to_index_future[ticker]
        frd_symbol = bbg_to_frd.get(index_future)
        if frd_symbol is None:
            print(f"Warning: No FRD symbol for index_future={index_future}")
            continue

        offset = offsets.get(exchange, "0min")
        cal = mcal.get_calendar(exchange)
        sched = cal.schedule(start_date=data_start, end_date=data_end)
        close_local = sched["market_close"].dt.tz_convert(str(cal.tz))
        normal_local_time = close_local.dt.time.mode()[0]
        sched = sched[close_local.dt.time == normal_local_time]
        close_times = sched["market_close"].dt.tz_convert("America/New_York") + pd.Timedelta(offset)

        futures_path = futures_dir / f"{frd_symbol}_full_1min_continuous_ratio_adjusted.txt"
        futures_at_close = sample_futures_at_close(futures_path, close_times)
        exchange_futures_at_close[exchange] = futures_at_close
        print(f"Computed futures-at-close for {exchange} using {frd_symbol}: {len(futures_at_close)} days")

    aligned_prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        idx_symbol = stock_to_index.get(ticker)
        if idx_symbol is None:
            continue

        exchange = stock_to_exchange.get(ticker)
        if ticker in misaligned_stocks and exchange in exchange_futures_at_close:
            aligned_prices[ticker] = exchange_futures_at_close[exchange]
        elif idx_symbol in index_data.columns:
            aligned_prices[ticker] = index_data[idx_symbol]

    result = pd.DataFrame(aligned_prices)
    result.index.name = "date"

    output_file = Path(output_path) if output_path is not None else resolve_repo_path(
        "data", "processed", "aligned_index_prices.csv", repo_root=repo_root
    )
    ensure_parent_dir(output_file)
    result.to_csv(output_file)
    print(f"Aligned index prices saved to {output_file}")
    print(f"Shape: {result.shape}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build ordinary-stock-aligned index prices.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--adr-info", default=None)
    parser.add_argument("--futures-symbols", default=None)
    parser.add_argument("--index-prices", default=None)
    parser.add_argument("--bloomberg-close-times", default=None)
    parser.add_argument("--close-offsets", default=None)
    parser.add_argument("--futures-minute-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    build_aligned_index_prices(
        adr_info_path=args.adr_info,
        futures_symbols_path=args.futures_symbols,
        index_prices_path=args.index_prices,
        bloomberg_close_times_path=args.bloomberg_close_times,
        close_offsets_path=args.close_offsets,
        futures_minute_dir=args.futures_minute_dir,
        output_path=args.output,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
