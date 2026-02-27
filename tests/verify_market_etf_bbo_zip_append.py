#!/usr/bin/env python3
"""
Verify that market ETF bbo-1m append from Databento ZIPs succeeded.

Checks:
1) exchange/ticker parquet targets exist
2) output schema matches existing exchange reference schema
3) date coverage includes every date present in ZIP payload files
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


DATE_RE = re.compile(r".*-(\d{8})\.bbo-1m\.dbn\.zst$")


def expected_dates_from_zip(zip_path: Path) -> set[str]:
    out: set[str] = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            m = DATE_RE.match(name)
            if m:
                out.add(pd.to_datetime(m.group(1), format="%Y%m%d").strftime("%Y-%m-%d"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Verify one-time market ETF BBO append output.")
    p.add_argument(
        "--output-dir",
        default="data/raw/etfs/market/bbo-1m/by_exchange",
        help="Parquet root to validate.",
    )
    p.add_argument(
        "--arcx-zip",
        default="ARCX-20260222-FLN6G65U95.zip",
        help="ARCX Databento ZIP used for append.",
    )
    p.add_argument(
        "--xnas-zip",
        default="XNAS-20260222-9Y7BVLF349.zip",
        help="XNAS Databento ZIP used for append.",
    )
    p.add_argument("--ticker", default="EEM", help="Ticker expected in appended output.")
    args = p.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Output root not found: {root}")

    targets = {
        "ARCX.PILLAR": root / "exchange=ARCX.PILLAR" / f"ticker={args.ticker}" / "data.parquet",
        "XNAS.ITCH": root / "exchange=XNAS.ITCH" / f"ticker={args.ticker}" / "data.parquet",
    }
    for ex, path in targets.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing target parquet for {ex}: {path}")

    ref_cols = {}
    for ex in targets:
        ref_file = next((root / f"exchange={ex}").glob("ticker=*/data.parquet"))
        ref_cols[ex] = pd.read_parquet(ref_file, engine="pyarrow").columns.tolist()

    for ex, path in targets.items():
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty:
            raise AssertionError(f"Appended parquet is empty: {path}")
        cols = df.columns.tolist()
        if cols != ref_cols[ex]:
            raise AssertionError(
                f"Schema mismatch for {ex} {args.ticker}.\n"
                f"expected={ref_cols[ex]}\n"
                f"actual={cols}"
            )

    expected = {
        "ARCX.PILLAR": expected_dates_from_zip(Path(args.arcx_zip)),
        "XNAS.ITCH": expected_dates_from_zip(Path(args.xnas_zip)),
    }
    for ex, path in targets.items():
        got_dates = set(pd.read_parquet(path, columns=["date"], engine="pyarrow")["date"].astype(str).unique().tolist())
        missing = sorted(expected[ex] - got_dates)
        if missing:
            raise AssertionError(f"{ex} missing {len(missing)} expected dates (first 10): {missing[:10]}")

    print("PASS: market ETF BBO ZIP append output verified.")
    for ex, path in targets.items():
        n = len(pd.read_parquet(path, columns=["date"], engine="pyarrow"))
        print(f"  {ex} {args.ticker}: {n} rows @ {path}")


if __name__ == "__main__":
    main()

