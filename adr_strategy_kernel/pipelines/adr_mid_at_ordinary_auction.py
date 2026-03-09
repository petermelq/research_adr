from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import polars as pl

from adr_strategy_kernel.common import ensure_parent_dir, get_market_business_days, load_adr_info, load_params, resolve_repo_path

DEFAULT_ORDINARY_AUCTION_OFFSETS = {
    "XLON": "6min",
    "XAMS": "6min",
    "XPAR": "6min",
    "XETR": "6min",
    "XMIL": "6min",
    "XBRU": "6min",
    "XMAD": "6min",
    "XHEL": "0min",
    "XDUB": "0min",
    "XOSL": "5min",
    "XSTO": "0min",
    "XSWX": "1min",
    "XCSE": "0min",
    "XASX": "11min",
    "XTKS": "1min",
}


def process_adr_mids_efficiently(
    adr_path: str | Path,
    ticker_close_df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    ticker_close_pl = pl.DataFrame(
        {
            "adr": ticker_close_df["adr"].tolist(),
            "exchange": ticker_close_df["exchange"].tolist(),
            "date": [pd.to_datetime(date_value) for date_value in ticker_close_df["date"].tolist()],
            "close_time": [pd.to_datetime(close_time) for close_time in ticker_close_df["close_time"].tolist()],
        }
    )

    adr_dir = Path(adr_path)
    ticker_dirs = [directory for directory in adr_dir.iterdir() if directory.is_dir() and directory.name.startswith("ticker=")]
    results: list[pl.DataFrame] = []

    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name.split("=")[1]
        ticker_close_filtered = ticker_close_pl.filter(pl.col("adr") == ticker)
        if ticker_close_filtered.height == 0:
            continue

        parquet_files = list(ticker_dir.rglob("*.parquet"))
        if not parquet_files:
            continue

        for parquet_file in parquet_files:
            try:
                adr_data = pl.read_parquet(parquet_file)
                if "date" not in adr_data.columns:
                    print(f"Warning: 'date' column not found in {parquet_file}, skipping")
                    continue

                adr_data = adr_data.with_columns(pl.col("date").cast(pl.Utf8).str.to_date().alias("date"))
                if start_date:
                    adr_data = adr_data.filter(pl.col("date") >= pd.to_datetime(start_date).date())
                if end_date:
                    adr_data = adr_data.filter(pl.col("date") <= pd.to_datetime(end_date).date())
                if adr_data.height == 0:
                    continue

                adr_data = adr_data.with_columns(((pl.col("nbbo_bid") + pl.col("nbbo_ask")) / 2).alias("mid"))
                adr_data = adr_data.select(["ts_recv", "mid", "Ticker", "date"])
                unique_dates = adr_data.select("date").unique().to_series().to_list()

                for partition_date in unique_dates:
                    date_data = adr_data.filter(pl.col("date") == partition_date)
                    if date_data.height == 0:
                        continue

                    date_close_filtered = ticker_close_filtered.filter(pl.col("date").dt.date() == partition_date)
                    if date_close_filtered.height == 0:
                        continue

                    date_close_converted = date_close_filtered.with_columns(
                        pl.col("close_time").dt.convert_time_zone("America/New_York").dt.cast_time_unit("ns")
                    )
                    joined_data = date_close_converted.join_asof(
                        date_data.sort("ts_recv"),
                        left_on="close_time",
                        right_on="ts_recv",
                    )
                    if joined_data.height > 0:
                        results.append(joined_data)
                        print(f"Processed {ticker} for {partition_date}: {joined_data.height} records")
            except Exception as exc:
                print(f"Error processing {ticker} in {parquet_file}: {exc}")

    if results:
        return pl.concat(results, how="vertical_relaxed")
    return pl.DataFrame()


cbday = get_market_business_days()


def get_daily_adj(adj_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    adj_df = adj_df.groupby("adjustment_date")[["adjustment_factor"]].prod().sort_index(ascending=False)
    adj_df.loc[start_date, "adjustment_factor"] = 1.0
    adj_df["cum_adj"] = adj_df["adjustment_factor"].cumprod()
    adj_df.index = [pd.to_datetime(idx) - cbday for idx in adj_df.index]
    adj_df.loc[end_date, "cum_adj"] = 1.0
    adj_df = adj_df.sort_index().loc[:end_date]
    return adj_df[["cum_adj"]].sort_index().resample("1D").bfill()


def build_ticker_close_frame(
    adr_info: pd.DataFrame,
    start_date: str,
    end_date: str,
    close_offsets: dict[str, str] | None = None,
) -> pd.DataFrame:
    close_offsets = close_offsets or DEFAULT_ORDINARY_AUCTION_OFFSETS
    exchanges = adr_info["exchange"].dropna().unique().tolist()
    close_time = pd.DataFrame(
        {
            exchange: mcal.get_calendar(exchange)
            .schedule(start_date=start_date, end_date=end_date)["market_close"]
            .dt.tz_convert("America/New_York")
            for exchange in exchanges
        }
    )
    for exchange, offset in close_offsets.items():
        if exchange in close_time.columns:
            close_time[exchange] += pd.Timedelta(offset)
    close_time = close_time.stack().reset_index(name="close_time").rename(
        columns={"level_0": "date", "level_1": "exchange"}
    )
    return adr_info[["adr_ticker", "exchange"]].rename(columns={"adr_ticker": "adr"}).merge(close_time, on="exchange")


def run_adr_mid_at_ordinary_auction(
    adr_path: str | Path | None = None,
    adr_info_path: str | Path | None = None,
    adjustment_factors_path: str | Path | None = None,
    params_path: str | Path | None = None,
    output_unadjusted_path: str | Path | None = None,
    output_adjusted_path: str | Path | None = None,
    close_offsets: dict[str, str] | None = None,
    write_adjusted_output: bool = True,
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adr_dir = Path(adr_path) if adr_path is not None else resolve_repo_path(
        "data", "raw", "adrs", "bbo-1m", "nbbo", repo_root=repo_root
    )
    output_unadjusted = Path(output_unadjusted_path) if output_unadjusted_path is not None else resolve_repo_path(
        "data", "processed", "adrs", "adr_mid_at_ord_auction_adjust_none.csv", repo_root=repo_root
    )
    output_adjusted = (
        Path(output_adjusted_path)
        if output_adjusted_path is not None
        else resolve_repo_path("data", "processed", "adrs", "adr_mid_at_ord_auction_adjust_all.csv", repo_root=repo_root)
    )

    params = load_params(params_path=params_path, repo_root=repo_root)
    start_date = params["start_date"]
    end_date = params["end_date"]
    adr_info = load_adr_info(adr_info_path=adr_info_path, repo_root=repo_root)

    ticker_close = build_ticker_close_frame(
        adr_info,
        start_date=start_date,
        end_date=end_date,
        close_offsets=close_offsets,
    )

    result_df = process_adr_mids_efficiently(
        adr_dir,
        ticker_close,
        start_date=start_date,
        end_date=end_date,
    )
    if result_df.is_empty():
        print("No data processed - check file paths and data availability")
        return pd.DataFrame(), pd.DataFrame()

    result_df = (
        result_df.pivot(on="adr", index="date", values="mid")
        .to_pandas()
        .set_index("date")
        .sort_index()
    )

    ensure_parent_dir(output_unadjusted)
    result_df.to_csv(output_unadjusted)
    print(f"Results saved to {output_unadjusted}")
    print(f"Processed {result_df.shape[0]} records")

    adj_result_df = pd.DataFrame()
    if write_adjusted_output:
        adjustment_file = Path(adjustment_factors_path) if adjustment_factors_path is not None else resolve_repo_path(
            "data", "processed", "adrs", "adr_adjustment_factors.csv", repo_root=repo_root
        )
        adj_factors = pd.read_csv(adjustment_file)
        adj_df = (
            adj_factors.groupby("ticker")
            .apply(get_daily_adj, start_date=start_date, end_date=end_date)
            .reset_index()
            .rename(columns={"level_1": "date"})
        )
        stacked_price = result_df.stack().reset_index(name="price").rename(columns={"level_1": "ticker"})
        adj_df["ticker"] = adj_df["ticker"].str.replace(" US Equity", "", regex=False)
        merged = stacked_price.merge(adj_df, on=["ticker", "date"], how="left")
        merged["adj_price"] = merged["price"] * merged["cum_adj"]
        adj_result_df = merged.pivot(index="date", columns="ticker", values="adj_price")
        ensure_parent_dir(output_adjusted)
        adj_result_df.to_csv(output_adjusted)

    return result_df, adj_result_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample ADR NBBO mids at the ordinary exchange closing auction.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--adr-path", default=None)
    parser.add_argument("--adr-info", default=None)
    parser.add_argument("--adjustment-factors", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--output-unadjusted", default=None)
    parser.add_argument("--output-adjusted", default=None)
    parser.add_argument("--skip-adjusted-output", action="store_true")
    args = parser.parse_args(argv)

    run_adr_mid_at_ordinary_auction(
        adr_path=args.adr_path,
        adr_info_path=args.adr_info,
        adjustment_factors_path=args.adjustment_factors,
        params_path=args.params,
        output_unadjusted_path=args.output_unadjusted,
        output_adjusted_path=args.output_adjusted,
        write_adjusted_output=not args.skip_adjusted_output,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
