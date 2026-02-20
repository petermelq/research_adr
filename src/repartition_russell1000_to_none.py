"""Convert Russell OHLCV date-partitioned files to per-ticker non-partitioned files.

This is a standalone maintenance script (not a DVC stage).
It merges `ticker=XXX/date=YYYY-MM-DD/data.parquet` into `ticker=XXX/data.parquet`.
"""

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/russell1000/ohlcv-1m"),
        help="Root Russell OHLCV directory.",
    )
    parser.add_argument(
        "--keep-date-partitions",
        action="store_true",
        help="Keep original date= directories after successful merge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing/deleting files.",
    )
    return parser.parse_args()


def merge_ticker_dir(
    ticker_dir: Path, keep_date_partitions: bool, dry_run: bool
) -> tuple[int, bool]:
    existing_flat = ticker_dir / "data.parquet"
    partition_files = []
    partition_files.extend(sorted(ticker_dir.glob("month=*/data.parquet")))
    partition_files.extend(sorted(ticker_dir.glob("month=*/date=*.parquet")))
    partition_files.extend(sorted(ticker_dir.glob("date=*/data.parquet")))
    partition_files.extend(sorted(ticker_dir.glob("date=*.parquet")))
    partition_files = sorted(set(partition_files))
    if not partition_files:
        return 0, False

    sources = []
    if existing_flat.exists():
        sources.append(existing_flat)
    sources.extend(partition_files)

    tmp_out = ticker_dir / "data.parquet.tmp"
    flat_out = ticker_dir / "data.parquet"

    if dry_run:
        return len(partition_files), True

    if tmp_out.exists():
        tmp_out.unlink()

    writer = None
    schema = None
    rows_written = 0

    def _align_to_schema(table: pa.Table, target: pa.Schema) -> pa.Table:
        cols = {}
        for field in target:
            name = field.name
            if name in table.column_names:
                col = table[name]
                if not col.type.equals(field.type):
                    col = col.cast(field.type)
            else:
                col = pa.nulls(table.num_rows, type=field.type)
            cols[name] = col
        return pa.table(cols, schema=target)

    try:
        for src in sources:
            table = pq.read_table(src)
            if table.num_rows == 0:
                continue
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp_out, schema=schema, compression="zstd")
            elif table.schema != schema:
                table = _align_to_schema(table, schema)
            writer.write_table(table)
            rows_written += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if rows_written == 0:
        if tmp_out.exists():
            tmp_out.unlink()
        return len(date_files), False

    tmp_out.replace(flat_out)

    if not keep_date_partitions:
        for f in partition_files:
            shutil.rmtree(f.parent, ignore_errors=True)

    return len(partition_files), True


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ticker_dirs = sorted(
        [p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("ticker=")]
    )

    converted = 0
    skipped = 0
    date_files_seen = 0
    for tdir in ticker_dirs:
        n_date_files, did_convert = merge_ticker_dir(
            tdir, keep_date_partitions=args.keep_date_partitions, dry_run=args.dry_run
        )
        date_files_seen += n_date_files
        if did_convert:
            converted += 1
        else:
            skipped += 1

    mode = "DRY RUN" if args.dry_run else "DONE"
    print(
        f"[{mode}] tickers={len(ticker_dirs)} converted={converted} skipped={skipped} "
        f"date_files_seen={date_files_seen}"
    )


if __name__ == "__main__":
    main()
