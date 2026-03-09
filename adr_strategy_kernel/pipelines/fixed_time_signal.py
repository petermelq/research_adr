from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from adr_strategy_kernel.common import ensure_parent_dir, load_adr_info, load_params, resolve_repo_path
from adr_strategy_kernel.pipelines.fixed_time_mid import load_ticker_list


def extract_fixed_time_signal(
    signal_dir: str | Path,
    tickers: list[str],
    time_to_save: dt.time,
    output_path: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    signal_df = pd.read_parquet(signal_dir)
    if start_date is not None:
        signal_df = signal_df[signal_df["date"].astype(str) >= pd.Timestamp(start_date).strftime("%Y-%m-%d")]
    if end_date is not None:
        signal_df = signal_df[signal_df["date"].astype(str) <= pd.Timestamp(end_date).strftime("%Y-%m-%d")]

    all_signal: dict[str, pd.Series] = {}
    for ticker in tickers:
        ticker_signal_df = signal_df[signal_df["ticker"] == ticker]
        if ticker_signal_df.empty:
            continue
        all_signal[ticker] = (
            ticker_signal_df.between_time("00:00", time_to_save)
            .groupby("date", observed=True)["signal"]
            .last()
        )

    result = pd.DataFrame(all_signal)
    result.index = pd.to_datetime(result.index)
    result = result.sort_index().sort_index(axis=1)
    if output_path is not None:
        output_file = Path(output_path)
        ensure_parent_dir(output_file)
        result.to_csv(output_file)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract fixed-time daily signals from partitioned minute signals.")
    parser.add_argument("signal_dir", nargs="?", default=None)
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--tickers-csv", default=None)
    parser.add_argument("--tickers-columns", nargs="+", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--time-to-save-hours", type=int, default=None)
    parser.add_argument("--time-to-save-minutes", type=int, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args(argv)

    signal_dir = Path(args.signal_dir) if args.signal_dir is not None else resolve_repo_path(
        "data", "processed", "futures_only_signal", repo_root=args.repo_root
    )
    output = Path(args.output) if args.output is not None else resolve_repo_path(
        "data", "processed", "fixed_time_signal.csv", repo_root=args.repo_root
    )
    params = load_params(params_path=args.params, repo_root=args.repo_root)
    hours = args.time_to_save_hours if args.time_to_save_hours is not None else params["fixed_trade_time_hours"]
    minutes = args.time_to_save_minutes if args.time_to_save_minutes is not None else params["fixed_trade_time_min"]
    start_date = args.start_date or params.get("start_date")
    end_date = args.end_date or params.get("end_date")

    if args.tickers or args.tickers_csv:
        tickers = load_ticker_list(
            tickers=args.tickers,
            tickers_csv=args.tickers_csv,
            tickers_columns=args.tickers_columns,
        )
    else:
        adr_info = load_adr_info(repo_root=args.repo_root)
        tickers = adr_info["adr_ticker"].tolist()

    extract_fixed_time_signal(
        signal_dir=signal_dir,
        tickers=tickers,
        time_to_save=dt.time(hours, minutes),
        output_path=output,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
