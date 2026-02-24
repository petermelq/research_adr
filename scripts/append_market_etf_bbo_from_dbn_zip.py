#!/usr/bin/env python3
"""
One-time utility to append Databento DBN ZIP downloads into an existing
partition_period=none parquet dataset.

This script is intended for cases where API download is unavailable but
Databento ZIP artifacts already exist locally.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

from data_tools.cli.batch_download_to_parquet import save_parquet
from data_tools.cli.batch_download_to_parquet_with_append import combine_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append Databento DBN ZIP files into market ETF bbo-1m parquet dataset."
    )
    p.add_argument(
        "--zip-files",
        nargs="+",
        required=True,
        help="Paths to Databento ZIP files (e.g. ARCX-...zip XNAS-...zip).",
    )
    p.add_argument(
        "--output-dir",
        default="data/raw/etfs/market/bbo-1m/by_exchange",
        help="Destination parquet root (default: data/raw/etfs/market/bbo-1m/by_exchange).",
    )
    p.add_argument(
        "--partition-period",
        default="none",
        choices=["none", "month", "year", "date"],
        help="Target partition period for combine step (default: none).",
    )
    p.add_argument(
        "--dataset-partition-name",
        default="exchange",
        help="Dataset partition key name (default: exchange).",
    )
    p.add_argument(
        "--n-date-group",
        type=int,
        default=50,
        help="Number of DBN day files to process per batch (default: 50).",
    )
    return p.parse_args()


def load_metadata_from_zip(zip_path: Path) -> tuple[str, str, str, list[str]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        md = json.loads(zf.read("metadata.json"))
    query = md.get("query", {})
    dataset = query.get("dataset")
    schema = query.get("schema")
    symbols = query.get("symbols", [])
    job_id = md.get("job_id")
    if not dataset or not schema or not job_id:
        raise ValueError(f"Missing required metadata fields in {zip_path}")
    return job_id, dataset, schema, symbols


def main() -> None:
    args = parse_args()
    zip_files = [Path(p).expanduser().resolve() for p in args.zip_files]
    for z in zip_files:
        if not z.exists():
            raise FileNotFoundError(f"ZIP file not found: {z}")

    job_ids: list[tuple[str, str]] = []
    datasets: list[str] = []
    tickers: list[str] = []
    schema_seen: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="dbn_zip_extract_") as extract_dir, tempfile.TemporaryDirectory(
        prefix="dbn_zip_parquet_"
    ) as temp_parquet:
        extract_root = Path(extract_dir)

        # Build a synthetic download layout expected by save_parquet:
        # <extract_root>/<job_id>/*.{schema}.dbn.zst
        for z in zip_files:
            job_id, dataset, schema, symbols = load_metadata_from_zip(z)
            schema_seen.add(schema)
            datasets.append(dataset)
            tickers.extend(symbols)
            job_ids.append((dataset, job_id))

            target_dir = extract_root / job_id
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(target_dir)

        if schema_seen != {"bbo-1m"}:
            raise ValueError(f"Expected only bbo-1m archives, got schemas: {sorted(schema_seen)}")

        # The conversion utility expects an argparse-like namespace.
        conv_args = SimpleNamespace(
            schema="bbo-1m",
            n_date_group=args.n_date_group,
            include_metadata=False,
            futures=False,
            dataset_partition_name=args.dataset_partition_name,
            start_date=None,
            end_date=None,
        )

        # Use the same converter/combiner as pipeline for schema/layout consistency.
        save_parquet(
            temp_dir=str(extract_root),
            output_dir=temp_parquet,
            job_ids=job_ids,
            client=None,  # unused in non-futures path
            args=conv_args,
            datasets=sorted(set(datasets)),
            tickers=sorted(set(tickers)),
        )
        combine_data(
            temp_parquet_with_dates=temp_parquet,
            dest_dir=args.output_dir,
            partition_period=args.partition_period,
        )

    print("Append complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {sorted(set(datasets))}")
    print(f"Tickers: {sorted(set(tickers))}")


if __name__ == "__main__":
    main()
