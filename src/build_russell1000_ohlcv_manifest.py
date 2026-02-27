from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def build_ticker_row(ticker_dir: Path) -> dict[str, object]:
    ticker = ticker_dir.name.split("=", 1)[1]
    file_count = 0
    total_bytes = 0
    max_mtime_ns = 0
    h = hashlib.sha256()

    for fp in sorted(ticker_dir.rglob("*.parquet")):
        st = fp.stat()
        rel = fp.relative_to(ticker_dir).as_posix()
        file_count += 1
        total_bytes += st.st_size
        if st.st_mtime_ns > max_mtime_ns:
            max_mtime_ns = st.st_mtime_ns
        h.update(f"{rel}\t{st.st_size}\t{st.st_mtime_ns}\n".encode("utf-8"))

    return {
        "ticker": ticker,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "max_mtime_ns": max_mtime_ns,
        "digest": h.hexdigest(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    root = args.input_dir
    ticker_dirs = sorted([d for d in root.glob("ticker=*") if d.is_dir()])
    rows = [build_ticker_row(td) for td in ticker_dirs]

    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote manifest for {len(df)} tickers -> {args.output}")


if __name__ == "__main__":
    main()
