from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def process_ticker_dir(ticker_dir: Path) -> dict[str, int]:
    stats = {
        "months_touched": 0,
        "date_dirs_removed": 0,
        "root_file_removed": 0,
        "files_moved": 0,
    }

    date_dirs = sorted([d for d in ticker_dir.glob("date=*") if d.is_dir()])
    if not date_dirs:
        return stats

    # If daily partitions exist, drop ticker-level combined file to avoid duplication.
    root_file = ticker_dir / "data.parquet"
    if root_file.exists():
        root_file.unlink()
        stats["root_file_removed"] = 1

    touched_months = set()
    for d in date_dirs:
        date_str = d.name.split("=", 1)[1]
        month_str = date_str[:7]
        src = d / "data.parquet"
        if src.exists():
            month_dir = ticker_dir / f"month={month_str}"
            month_dir.mkdir(parents=True, exist_ok=True)
            dst = month_dir / f"date={date_str}.parquet"
            # Move only if target does not exist; otherwise treat source as duplicate.
            if not dst.exists():
                shutil.move(str(src), str(dst))
                stats["files_moved"] += 1
            else:
                src.unlink()
            touched_months.add(month_str)
        shutil.rmtree(d, ignore_errors=True)
        stats["date_dirs_removed"] += 1

    stats["months_touched"] = len(touched_months)

    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/russell1000/ohlcv-1m"),
        help="Root directory with ticker=* partitions",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    args = p.parse_args()

    root = args.input_dir
    ticker_dirs = sorted([d for d in root.glob("ticker=*") if d.is_dir()])
    print(f"Found {len(ticker_dirs)} ticker directories in {root}")

    totals = {
        "tickers": 0,
        "months_touched": 0,
        "date_dirs_removed": 0,
        "root_files_removed": 0,
        "files_moved": 0,
    }

    if args.workers <= 1:
        for i, tdir in enumerate(ticker_dirs, start=1):
            s = process_ticker_dir(tdir)
            totals["tickers"] += 1
            totals["months_touched"] += s["months_touched"]
            totals["date_dirs_removed"] += s["date_dirs_removed"]
            totals["root_files_removed"] += s["root_file_removed"]
            totals["files_moved"] += s["files_moved"]
            if i % 50 == 0 or i == len(ticker_dirs):
                print(
                    f"[{i}/{len(ticker_dirs)}] months_touched={totals['months_touched']} "
                    f"date_dirs_removed={totals['date_dirs_removed']} "
                    f"root_files_removed={totals['root_files_removed']} "
                    f"files_moved={totals['files_moved']}"
                )
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_ticker_dir, tdir): tdir for tdir in ticker_dirs}
            for fut in as_completed(futs):
                s = fut.result()
                done += 1
                totals["tickers"] += 1
                totals["months_touched"] += s["months_touched"]
                totals["date_dirs_removed"] += s["date_dirs_removed"]
                totals["root_files_removed"] += s["root_file_removed"]
                totals["files_moved"] += s["files_moved"]
                if done % 50 == 0 or done == len(ticker_dirs):
                    print(
                        f"[{done}/{len(ticker_dirs)}] months_touched={totals['months_touched']} "
                        f"date_dirs_removed={totals['date_dirs_removed']} "
                        f"root_files_removed={totals['root_files_removed']} "
                        f"files_moved={totals['files_moved']}"
                    )

    print("Done.")
    print(totals)


if __name__ == "__main__":
    main()
