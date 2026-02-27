#!/usr/bin/env python3
"""One-time safe parallel repartition from ticker/date=* shards to ticker/data.parquet.

Safety guarantees:
- Writes each ticker output to a temporary parquet and atomically replaces data.parquet.
- Deletes date partitions only after a successful atomic replace.
- Resumable/idempotent: already-converted tickers (no date= dirs) are skipped.
"""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class RepartitionResult:
    ticker: str
    status: str
    rows: int = 0
    date_partitions: int = 0
    message: str = ""


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if "__index_level_0__" in df.columns:
        out = df.copy()
        out["__index_level_0__"] = pd.to_datetime(out["__index_level_0__"], errors="coerce")
        out = out.dropna(subset=["__index_level_0__"]).set_index("__index_level_0__")
        return out
    if "DateTime" in df.columns:
        out = df.copy()
        out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
        out = out.dropna(subset=["DateTime"]).set_index("DateTime")
        return out
    raise ValueError("could not infer datetime index")


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return _ensure_datetime_index(df)


def repartition_one_ticker(ticker_dir: Path, compression: str) -> RepartitionResult:
    ticker = ticker_dir.name.split("=", 1)[-1]
    date_dirs = sorted(Path(p) for p in glob(str(ticker_dir / "date=*")) if Path(p).is_dir())
    if not date_dirs:
        return RepartitionResult(ticker=ticker, status="skipped_no_date")

    frames: List[pd.DataFrame] = []
    existing_out = ticker_dir / "data.parquet"
    if existing_out.exists():
        frames.append(_read_parquet(existing_out))

    kept_date_dirs: List[Path] = []
    for d in date_dirs:
        p = d / "data.parquet"
        if not p.exists():
            continue
        frames.append(_read_parquet(p))
        kept_date_dirs.append(d)

    if not frames:
        return RepartitionResult(ticker=ticker, status="skipped_empty", date_partitions=len(date_dirs))

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    tmp_path = ticker_dir / f".data.parquet.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        combined.to_parquet(tmp_path, compression=compression)
        os.replace(tmp_path, existing_out)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    # Delete date partitions only after successful atomic replace.
    for d in kept_date_dirs:
        shutil.rmtree(d)

    return RepartitionResult(
        ticker=ticker,
        status="converted",
        rows=len(combined),
        date_partitions=len(kept_date_dirs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely repartition ticker/date=* to ticker/data.parquet in parallel.")
    parser.add_argument("--input-dir", default="data/raw/russell1000/ohlcv-1m")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compression", default="brotli")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    ticker_dirs = sorted(p for p in input_dir.glob("ticker=*") if p.is_dir())
    if not ticker_dirs:
        print(f"No ticker directories found in {input_dir}")
        return

    print(f"Found {len(ticker_dirs)} ticker directories")
    print(f"Running safe repartition with workers={args.workers}, compression={args.compression}")

    converted = 0
    skipped = 0
    failed = 0
    rows_total = 0
    dates_total = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_ticker = {
            ex.submit(repartition_one_ticker, ticker_dir, args.compression): ticker_dir.name
            for ticker_dir in ticker_dirs
        }
        for i, fut in enumerate(as_completed(fut_to_ticker), start=1):
            ticker_name = fut_to_ticker[fut]
            try:
                res = fut.result()
                if res.status == "converted":
                    converted += 1
                    rows_total += res.rows
                    dates_total += res.date_partitions
                else:
                    skipped += 1
                if i % 25 == 0 or res.status == "converted":
                    print(
                        f"[{i}/{len(ticker_dirs)}] {ticker_name}: {res.status} "
                        f"(converted={converted}, skipped={skipped}, failed={failed})"
                    )
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(ticker_dirs)}] {ticker_name}: FAILED: {e}")

    print("Done.")
    print(
        f"Summary: converted={converted}, skipped={skipped}, failed={failed}, "
        f"date_partitions_removed={dates_total}, rows_written={rows_total}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
