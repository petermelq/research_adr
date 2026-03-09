from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from adr_strategy_kernel.common import ensure_parent_dir, load_params, resolve_repo_path, strip_us_equity_suffix


def load_ticker_list(
    tickers: list[str] | None = None,
    tickers_csv: str | Path | None = None,
    tickers_columns: list[str] | None = None,
) -> list[str]:
    if tickers:
        return sorted({strip_us_equity_suffix(value) for value in tickers if isinstance(value, str) and value})
    if tickers_csv is None:
        raise ValueError("Either tickers or tickers_csv must be provided")

    tickers_df = pd.read_csv(tickers_csv)
    columns = tickers_columns or ["ticker"]
    values: list[str] = []
    for column in columns:
        if column not in tickers_df.columns:
            raise KeyError(f"Ticker column {column!r} not found in {tickers_csv}")
        values.extend(tickers_df[column].dropna().astype(str).tolist())
    return sorted({strip_us_equity_suffix(value) for value in values if value})


def extract_daily_fixed_time_mid(
    nbbo_dir: str | Path,
    time_to_save: dt.time,
    tickers: list[str],
    output_path: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None
    ny_close_times = (
        mcal.get_calendar("XNYS")
        .schedule(
            start_date=(start_ts or pd.Timestamp("1980-01-01")).strftime("%Y-%m-%d"),
            end_date=(end_ts or pd.Timestamp("2030-01-01")).strftime("%Y-%m-%d"),
        )["market_close"]
        .dt.tz_convert("America/New_York")
    )
    ny_close_times.index = ny_close_times.index.astype(str)
    start_time = (dt.datetime.combine(dt.date.today(), time_to_save) - pd.Timedelta("30min")).time()

    all_mid: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = pd.read_parquet(
            nbbo_dir,
            filters=[("ticker", "==", ticker)],
            columns=["nbbo_bid", "nbbo_ask", "date"],
        )
        if df.empty:
            continue
        if start_ts is not None:
            df = df[df["date"].astype(str) >= start_ts.strftime("%Y-%m-%d")]
        if end_ts is not None:
            df = df[df["date"].astype(str) <= end_ts.strftime("%Y-%m-%d")]
        if df.empty:
            continue

        df["mid"] = (df["nbbo_bid"] + df["nbbo_ask"]) / 2
        df = df.merge(ny_close_times, left_on="date", right_index=True)
        df = df[df["market_close"].dt.time == dt.time(16, 0)]
        df = df.between_time(start_time, time_to_save)
        all_mid[ticker] = df.groupby("date")["mid"].last()

    mid_df = pd.DataFrame(all_mid)
    mid_df.index = pd.to_datetime(mid_df.index)
    mid_df = mid_df.sort_index().sort_index(axis=1)
    if output_path is not None:
        output_file = Path(output_path)
        ensure_parent_dir(output_file)
        mid_df.to_csv(output_file)
    return mid_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract daily fixed-time NBBO mids.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--nbbo-dir", default=None)
    parser.add_argument("--tickers-csv", default=None)
    parser.add_argument("--tickers-columns", nargs="+", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--time-to-save-hours", type=int, default=None)
    parser.add_argument("--time-to-save-minutes", type=int, default=0)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    nbbo_dir = Path(args.nbbo_dir) if args.nbbo_dir is not None else resolve_repo_path(
        "data", "raw", "adrs", "bbo-1m", "nbbo", repo_root=args.repo_root
    )
    params = load_params(params_path=args.params, repo_root=args.repo_root)
    time_to_save_hours = args.time_to_save_hours
    time_to_save_minutes = args.time_to_save_minutes
    if time_to_save_hours is None:
        time_to_save_hours = params["fixed_trade_time_hours"]
        time_to_save_minutes = params["fixed_trade_time_min"]
    start_date = args.start_date or params.get("start_date")
    end_date = args.end_date or params.get("end_date")
    tickers = load_ticker_list(
        tickers=args.tickers,
        tickers_csv=args.tickers_csv,
        tickers_columns=args.tickers_columns,
    )
    extract_daily_fixed_time_mid(
        nbbo_dir=nbbo_dir,
        time_to_save=dt.time(time_to_save_hours, time_to_save_minutes),
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
