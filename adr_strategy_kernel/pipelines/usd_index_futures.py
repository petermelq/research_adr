from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from adr_strategy_kernel.common import normalize_currency, resolve_repo_path

NOTIONAL_MULTIPLIERS = {
    "FTUK": 10,
    "FDAX": 25,
    "FCE": 10,
    "FXXP": 50,
    "FESX": 10,
    "FTI": 200,
    "NIY": 500,
    "MME": 50,
}


def load_fx_frame(
    currency: str,
    fx_dir: str | Path,
    fx_cache: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    if currency in fx_cache:
        return fx_cache[currency]

    fx_file = Path(fx_dir) / f"{currency}USD_full_1min.txt"
    fx_df = pl.read_csv(
        fx_file,
        has_header=False,
        new_columns=["date", "time", "open", "high", "low", "close", "volume"],
    )
    fx_cache[currency] = fx_df.with_columns(
        pl.concat_str(
            [
                pl.col("date").cast(pl.Utf8),
                pl.lit(" "),
                pl.col("time").cast(pl.Utf8),
            ]
        )
        .str.to_datetime(format="%Y%m%d %H:%M:%S")
        .dt.replace_time_zone("America/New_York")
        .alias("timestamp")
    ).select(["timestamp", pl.col("close").alias("fx_rate")])
    return fx_cache[currency]


def convert_index_futures_to_usd(
    futures_symbols_path: str | Path | None = None,
    futures_minute_dir: str | Path | None = None,
    fx_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    symbols: list[str] | None = None,
    repo_root: str | Path | None = None,
) -> list[Path]:
    futures_symbols_file = Path(futures_symbols_path) if futures_symbols_path is not None else resolve_repo_path(
        "data", "raw", "futures_symbols.csv", repo_root=repo_root
    )
    futures_symbols = pd.read_csv(futures_symbols_file)
    futures_symbols["first_rate_symbol"] = futures_symbols["first_rate_symbol"].astype(str).str.strip()
    futures_symbols["currency"] = futures_symbols["currency"].apply(normalize_currency).fillna("USD")
    symbol_to_currency = dict(zip(futures_symbols["first_rate_symbol"], futures_symbols["currency"]))

    futures_dir = Path(futures_minute_dir) if futures_minute_dir is not None else resolve_repo_path(
        "data", "raw", "futures", "minute_bars", repo_root=repo_root
    )
    fx_base = Path(fx_dir) if fx_dir is not None else resolve_repo_path(
        "data", "raw", "currencies", "minute_bars", repo_root=repo_root
    )
    output_base = Path(output_dir) if output_dir is not None else resolve_repo_path(
        "data", "processed", "futures", "converted_minute_bars", repo_root=repo_root
    )

    fx_cache: dict[str, pl.DataFrame] = {}
    output_files: list[Path] = []
    target_symbols = symbols or list(NOTIONAL_MULTIPLIERS)

    for futures_symbol in target_symbols:
        notional_multiplier = NOTIONAL_MULTIPLIERS[futures_symbol]
        currency = symbol_to_currency.get(futures_symbol, "USD")
        futures_file = futures_dir / f"{futures_symbol}_full_1min_continuous_ratio_adjusted.txt"
        output_file = output_base / f"symbol={futures_symbol}" / f"{futures_symbol}_close_to_usd_1min.parquet"

        futures_df = pl.read_csv(
            futures_file,
            has_header=False,
            new_columns=["timestamp", "open", "high", "low", "close", "volume"],
        ).with_columns(
            pl.col("timestamp")
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S")
            .dt.replace_time_zone("America/New_York"),
            (pl.col(["open", "high", "low", "close"]) * notional_multiplier),
        )

        if currency != "USD":
            futures_df = futures_df.join(
                load_fx_frame(currency, fx_base, fx_cache),
                on="timestamp",
                how="left",
            )
            futures_df = futures_df.with_columns(
                pl.col(["open", "high", "low", "close"]) * pl.col("fx_rate")
            )
        else:
            futures_df = futures_df.with_columns(pl.lit(1.0).alias("fx_rate"))

        output_file.parent.mkdir(parents=True, exist_ok=True)
        futures_df.write_parquet(output_file)
        output_files.append(output_file)
        print(f"Converted {futures_symbol} minute bars to USD at {output_file}")

    return output_files


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert FRD index futures minute bars to USD notionals.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--futures-symbols", default=None)
    parser.add_argument("--futures-minute-dir", default=None)
    parser.add_argument("--fx-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args(argv)

    convert_index_futures_to_usd(
        futures_symbols_path=args.futures_symbols,
        futures_minute_dir=args.futures_minute_dir,
        fx_dir=args.fx_dir,
        output_dir=args.output_dir,
        symbols=args.symbols,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
