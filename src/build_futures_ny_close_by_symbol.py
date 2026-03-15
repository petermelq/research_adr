import argparse
import os
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def _coerce_ny_ns(index_like) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index_like)
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    else:
        idx = idx.tz_convert("America/New_York")
    return idx.as_unit("ns")


def load_future_symbol_map(adr_info_path: Path, futures_symbols_path: Path) -> list[str]:
    adr_info = pd.read_csv(adr_info_path)
    adr_info["index_future_bbg"] = adr_info["index_future_bbg"].astype(str).str.strip()

    futures_symbols = pd.read_csv(futures_symbols_path)
    futures_symbols["bloomberg_symbol"] = futures_symbols["bloomberg_symbol"].astype(str).str.strip()
    futures_symbols["first_rate_symbol"] = futures_symbols["first_rate_symbol"].astype(str).str.strip()

    merged = adr_info.merge(
        futures_symbols[["bloomberg_symbol", "first_rate_symbol"]],
        left_on="index_future_bbg",
        right_on="bloomberg_symbol",
        how="left",
    )
    return sorted(sym for sym in merged["first_rate_symbol"].dropna().astype(str).unique().tolist() if sym)


def compute_symbol_ny_close(symbol: str, futures_dir: Path) -> pd.Series:
    path = futures_dir / f"symbol={symbol}" / f"{symbol}_close_to_usd_1min.parquet"
    if not path.exists():
        return pd.Series(dtype=float, name=symbol)

    df = pd.read_parquet(path, columns=["timestamp", "close"]).dropna(subset=["close"])
    if df.empty:
        return pd.Series(dtype=float, name=symbol)

    df["timestamp"] = _coerce_ny_ns(df["timestamp"])

    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
    ny_close = (
        df[df["timestamp"].dt.time <= pd.Timestamp("16:00").time()]
        .groupby("date")["close"]
        .last()
        .rename(symbol)
    )
    return ny_close


def main():
    parser = argparse.ArgumentParser(description="Build daily NY-close futures USD notionals by future symbol.")
    parser.add_argument(
        "--adr-info",
        default=str(SCRIPT_DIR / ".." / "data" / "raw" / "adr_info.csv"),
    )
    parser.add_argument(
        "--futures-symbols",
        default=str(SCRIPT_DIR / ".." / "data" / "raw" / "futures_symbols.csv"),
    )
    parser.add_argument(
        "--futures-dir",
        default=str(SCRIPT_DIR / ".." / "data" / "processed" / "futures" / "converted_minute_bars"),
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / ".." / "data" / "processed" / "futures" / "futures_usd_notional_ny_close_by_symbol.csv"),
    )
    args = parser.parse_args()

    adr_info_path = Path(args.adr_info).resolve()
    futures_symbols_path = Path(args.futures_symbols).resolve()
    futures_dir = Path(args.futures_dir).resolve()
    output_path = Path(args.output).resolve()

    symbols = load_future_symbol_map(adr_info_path, futures_symbols_path)
    ny_close = {}
    for symbol in symbols:
        series = compute_symbol_ny_close(symbol, futures_dir)
        if len(series) > 0:
            ny_close[symbol] = series
            print(f"Processed {symbol}: {len(series):,} rows")
        else:
            print(f"Skipped {symbol}: no data")

    out_df = pd.DataFrame(ny_close).sort_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
