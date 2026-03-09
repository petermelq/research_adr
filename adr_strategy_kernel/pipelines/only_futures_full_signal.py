from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from adr_strategy_kernel.common import (
    ASIA_EXCHANGES,
    ensure_parent_dir,
    load_adr_info,
    load_close_offsets,
    load_params,
    resolve_repo_path,
)


def prepare_adr_baseline(
    adr_info: pd.DataFrame,
    adr_domestic_close: pd.DataFrame,
    ord_close_to_usd: pd.DataFrame,
) -> pd.DataFrame:
    adr_dict = dict(zip(adr_info["id"], adr_info["adr_ticker"]))
    ord_close_to_usd = ord_close_to_usd.rename(columns=adr_dict)
    asia_tickers = adr_info.loc[adr_info["exchange"].isin(ASIA_EXCHANGES), "adr_ticker"].tolist()
    available_asia_tickers = [ticker for ticker in asia_tickers if ticker in ord_close_to_usd.columns]
    asia_baseline = ord_close_to_usd[available_asia_tickers]
    return pd.concat(
        [
            adr_domestic_close.drop(columns=available_asia_tickers, errors="ignore"),
            asia_baseline,
        ],
        axis=1,
    ).sort_index()


def compute_futures_close_before_offset(
    merged_fut: pd.DataFrame,
    offset: pd.Timedelta,
) -> pd.DataFrame:
    return merged_fut.groupby("date")[["domestic_close_time", "close"]].apply(
        lambda frame: frame[frame.index <= frame["domestic_close_time"] + offset].iloc[-1]["close"]
        if (frame.index <= frame["domestic_close_time"] + offset).any()
        else np.nan
    ).to_frame(name="fut_domestic_close")


def run_only_futures_full_signal(
    futures_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    domestic_close_mid_path: str | Path | None = None,
    ord_close_to_usd_path: str | Path | None = None,
    betas_path: str | Path | None = None,
    adr_nbbo_dir: str | Path | None = None,
    adr_info_path: str | Path | None = None,
    futures_symbols_path: str | Path | None = None,
    close_offsets_path: str | Path | None = None,
    params_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> None:
    futures_dir = Path(futures_dir) if futures_dir is not None else resolve_repo_path(
        "data", "processed", "futures", "converted_minute_bars", repo_root=repo_root
    )
    output_dir = Path(output_dir) if output_dir is not None else resolve_repo_path(
        "data", "processed", "futures_only_signal", repo_root=repo_root
    )
    domestic_close_mid_file = Path(domestic_close_mid_path) if domestic_close_mid_path is not None else resolve_repo_path(
        "data", "processed", "adrs", "adr_mid_at_ord_auction_adjust_none.csv", repo_root=repo_root
    )
    ord_close_to_usd_file = Path(ord_close_to_usd_path) if ord_close_to_usd_path is not None else resolve_repo_path(
        "data", "processed", "ordinary", "ord_close_to_usd_adr_PX_LAST_adjust_none.csv", repo_root=repo_root
    )
    betas_file = Path(betas_path) if betas_path is not None else resolve_repo_path(
        "data", "processed", "models", "ordinary_betas_index_only.csv", repo_root=repo_root
    )
    adr_nbbo_dir = Path(adr_nbbo_dir) if adr_nbbo_dir is not None else resolve_repo_path(
        "data", "raw", "adrs", "bbo-1m", "nbbo", repo_root=repo_root
    )
    futures_symbols_file = Path(futures_symbols_path) if futures_symbols_path is not None else resolve_repo_path(
        "data", "raw", "futures_symbols.csv", repo_root=repo_root
    )
    close_offsets_file = Path(close_offsets_path) if close_offsets_path is not None else resolve_repo_path(
        "data", "raw", "close_time_offsets.csv", repo_root=repo_root
    )

    adr_info = load_adr_info(adr_info_path=adr_info_path, repo_root=repo_root)
    adr_tickers = adr_info["adr_ticker"].tolist()
    exchange_dict = adr_info.set_index("adr_ticker")["exchange"].to_dict()

    adr_domestic_close = pd.read_csv(domestic_close_mid_file, index_col=0)
    ord_close_to_usd = pd.read_csv(ord_close_to_usd_file, index_col=0)
    adr_domestic_close = prepare_adr_baseline(adr_info, adr_domestic_close, ord_close_to_usd)

    params = load_params(params_path=params_path, repo_root=repo_root)
    start_date = params["start_date"]
    end_date = params["end_date"]

    futures_df = pd.read_parquet(
        futures_dir,
        filters=[("timestamp", ">=", pd.Timestamp(start_date, tz="America/New_York"))],
        columns=["timestamp", "symbol", "close"],
    )
    futures_df["date"] = futures_df["timestamp"].dt.strftime("%Y-%m-%d")
    futures_df = futures_df.set_index("timestamp")

    time_futures_after_close = load_close_offsets(close_offsets_file)
    betas = pd.read_csv(betas_file, index_col=0)

    exchanges = adr_info["exchange"].dropna().unique().tolist()
    missing_offsets = sorted(set(exchanges) - set(time_futures_after_close))
    if missing_offsets:
        raise RuntimeError(
            "Missing close-time offsets for exchanges in data/raw/close_time_offsets.csv: "
            + ", ".join(missing_offsets)
        )

    close_times = {}
    for exchange in exchanges:
        close_times[exchange] = (
            mcal.get_calendar(exchange)
            .schedule(start_date=start_date, end_date=end_date)["market_close"]
            .dt.tz_convert("America/New_York")
        ).rename("domestic_close_time")
        close_times[exchange].index = close_times[exchange].index.astype(str)

    futures_symbols = pd.read_csv(futures_symbols_file)
    futures_symbols["bloomberg_symbol"] = futures_symbols["bloomberg_symbol"].astype(str).str.strip()
    futures_symbols["first_rate_symbol"] = futures_symbols["first_rate_symbol"].astype(str).str.strip()
    stock_to_index = adr_info.merge(
        futures_symbols,
        left_on="index_future_bbg",
        right_on="bloomberg_symbol",
        how="left",
    ).set_index("adr_ticker")["first_rate_symbol"].to_dict()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ticker in adr_tickers:
        exchange = exchange_dict[ticker]
        close_df = close_times[exchange]
        futures_symbol = stock_to_index.get(ticker)
        if not isinstance(futures_symbol, str) or futures_symbol == "":
            print(f"No futures mapping for {ticker}, skipping...")
            continue

        ticker_futures = futures_df[futures_df["symbol"] == futures_symbol].copy()
        if ticker_futures.empty:
            print(f"No futures data for {ticker} ({futures_symbol}), skipping...")
            continue
        merged_fut = ticker_futures.merge(close_df, left_on="date", right_index=True)

        adr_df = pd.read_parquet(
            adr_nbbo_dir,
            filters=[("ticker", "==", ticker)],
            columns=["ticker", "date", "nbbo_bid", "nbbo_ask"],
        )
        if adr_df.empty:
            print(f"No ADR NBBO data for {ticker}, skipping...")
            continue
        if ticker not in adr_domestic_close.columns:
            print(f"No domestic-close baseline for {ticker}, skipping...")
            continue

        adr_df["mid"] = (adr_df["nbbo_bid"] + adr_df["nbbo_ask"]) / 2
        merged_adr = adr_df.merge(
            adr_domestic_close[ticker].rename("adr_domestic_close"),
            left_on="date",
            right_index=True,
        )
        merged_adr["adr_ret"] = (
            (merged_adr["mid"] - merged_adr["adr_domestic_close"]) / merged_adr["adr_domestic_close"]
        ).to_frame(name="adr_ret")
        merged_adr = merged_adr.merge(close_df, left_on="date", right_index=True)
        adr_ret = merged_adr[
            merged_adr.index >= merged_adr["domestic_close_time"] + time_futures_after_close[exchange]
        ]["adr_ret"]

        fut_domestic_close = compute_futures_close_before_offset(
            merged_fut,
            time_futures_after_close[exchange],
        )
        merged_fut_after_close = merged_fut[merged_fut.index > merged_fut["domestic_close_time"]].copy()
        merged_fut_after_close = merged_fut_after_close.merge(fut_domestic_close, left_on="date", right_index=True)
        merged_fut_after_close["fut_ret"] = (
            (merged_fut_after_close["close"] - merged_fut_after_close["fut_domestic_close"])
            / merged_fut_after_close["fut_domestic_close"]
        )

        if ticker not in betas.columns:
            print(f"No beta for {ticker}, skipping...")
            continue
        merged_fut_after_close = merged_fut_after_close.merge(
            betas[ticker].rename("beta"),
            left_on="date",
            right_index=True,
        )

        merged_all = merged_fut_after_close.merge(merged_adr[["adr_ret"]], left_index=True, right_index=True)
        merged_all["signal"] = merged_all["fut_ret"] * merged_all["beta"] - merged_all["adr_ret"]
        merged_all["date"] = merged_all.index.strftime("%Y-%m-%d")
        signal_df = (
            merged_all[["signal", "date"]]
            .copy()
            .groupby("date")[["signal"]]
            .apply(lambda frame: frame[["signal"]].resample("1min").first().ffill())
            .droplevel(0)
        )
        signal_df["date"] = signal_df.index.strftime("%Y-%m-%d")

        output_file = output_dir / f"ticker={ticker}" / "data.parquet"
        ensure_parent_dir(output_file)
        signal_df.to_parquet(output_file)
        print(f"Processed and saved signal for {ticker}")

    print(f"Saved partitioned parquet dataset to {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Process futures and ADR data to compute futures-only signals.")
    parser.add_argument("futures_dir", nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--domestic-close-mid", default=None)
    parser.add_argument("--ord-close-to-usd", default=None)
    parser.add_argument("--betas", default=None)
    parser.add_argument("--adr-nbbo-dir", default=None)
    parser.add_argument("--adr-info", default=None)
    parser.add_argument("--futures-symbols", default=None)
    parser.add_argument("--close-offsets", default=None)
    parser.add_argument("--params", default=None)
    args = parser.parse_args(argv)

    run_only_futures_full_signal(
        futures_dir=args.futures_dir,
        output_dir=args.output_dir,
        domestic_close_mid_path=args.domestic_close_mid,
        ord_close_to_usd_path=args.ord_close_to_usd,
        betas_path=args.betas,
        adr_nbbo_dir=args.adr_nbbo_dir,
        adr_info_path=args.adr_info,
        futures_symbols_path=args.futures_symbols,
        close_offsets_path=args.close_offsets,
        params_path=args.params,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
