#!/usr/bin/env python3
"""Standalone parity check: 30m signal must match full-minute signal at eval times."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EVAL_TIMES = {"13:00", "13:30", "14:00", "14:30", "15:00", "15:30"}


def _load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["signal", "date"])
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex in {path}")
    return df.sort_index()


def _eval_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.index.strftime("%H:%M").isin(EVAL_TIMES)].copy()


def compare_dirs(full_dir: Path, sparse_dir: Path, atol: float = 1e-4, rtol: float = 1e-6) -> None:
    full_files = {p.parent.name.replace("ticker=", ""): p for p in full_dir.glob("ticker=*/data.parquet")}
    sparse_files = {p.parent.name.replace("ticker=", ""): p for p in sparse_dir.glob("ticker=*/data.parquet")}

    if not full_files:
        raise ValueError(f"No ticker parquet files found in {full_dir}")
    if not sparse_files:
        raise ValueError(f"No ticker parquet files found in {sparse_dir}")

    only_full = sorted(set(full_files) - set(sparse_files))
    only_sparse = sorted(set(sparse_files) - set(full_files))
    if only_full or only_sparse:
        msg = []
        if only_full:
            msg.append(f"Tickers only in full: {only_full[:10]}{' ...' if len(only_full) > 10 else ''}")
        if only_sparse:
            msg.append(f"Tickers only in 30m: {only_sparse[:10]}{' ...' if len(only_sparse) > 10 else ''}")
        raise AssertionError("; ".join(msg))

    tickers = sorted(full_files.keys())
    mismatches = []
    total_rows = 0
    non_empty = 0

    for ticker in tickers:
        full_df = _eval_slice(_load_signal(full_files[ticker]))
        sparse_df = _load_signal(sparse_files[ticker])

        if not full_df.index.equals(sparse_df.index):
            mismatches.append(f"{ticker}: index mismatch (full={len(full_df)} rows, sparse={len(sparse_df)} rows)")
            continue
        if not full_df["date"].equals(sparse_df["date"]):
            mismatches.append(f"{ticker}: date column mismatch")
            continue

        x = full_df["signal"].to_numpy(dtype=np.float64)
        y = sparse_df["signal"].to_numpy(dtype=np.float64)
        if not np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True):
            max_abs = float(np.nanmax(np.abs(x - y)))
            mismatches.append(f"{ticker}: signal mismatch (max_abs_diff={max_abs:.6g})")
            continue

        total_rows += len(sparse_df)
        if len(sparse_df) > 0:
            non_empty += 1

    if mismatches:
        preview = "\n".join(mismatches[:20])
        extra = f"\n... and {len(mismatches)-20} more" if len(mismatches) > 20 else ""
        raise AssertionError(f"Parity check failed for {len(mismatches)}/{len(tickers)} tickers:\n{preview}{extra}")

    print(
        "PASS: 30m signal matches full-minute signal at eval times "
        f"for {len(tickers)} tickers (non_empty={non_empty}, rows={total_rows})."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-dir", type=Path, required=True, help="Directory of full-minute signal output.")
    p.add_argument("--sparse-dir", type=Path, required=True, help="Directory of 30m-only signal output.")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compare_dirs(args.full_dir, args.sparse_dir, atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
