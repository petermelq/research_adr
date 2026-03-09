from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import REPO_ROOT, derive_required_inputs, load_artifact_tables, load_manifest


def run_import(source_repo: str, source_path: str, target_path: str, rev: str | None) -> None:
    command = ["dvc", "import", "--force"]
    if rev:
        command.extend(["--rev", rev])
    command.extend([source_repo, source_path, target_path])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate model_build imports via dvc import.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--imports-root", default="model_build/imports")
    args = parser.parse_args()

    manifest = load_manifest(args.artifact_dir)
    tables = load_artifact_tables(args.artifact_dir)
    derived = derive_required_inputs(tables["adr_info"], tables["futures_symbols"])

    source_repo = str(manifest["sources"]["dvc_repo"])
    source_rev = manifest["sources"].get("dvc_rev")
    imports_root = Path(args.imports_root)

    for ticker in derived["adr_tickers"]["ticker"].tolist():
        run_import(
            source_repo,
            f"data/raw/adrs/bbo-1m/nbbo/ticker={ticker}",
            str(imports_root / "data" / "raw" / "adrs" / "bbo-1m" / "nbbo" / f"ticker={ticker}"),
            source_rev,
        )

    for ticker in derived["market_etf_tickers"]["ticker"].tolist():
        run_import(
            source_repo,
            f"data/raw/etfs/market/bbo-1m/nbbo/ticker={ticker}",
            str(imports_root / "data" / "raw" / "etfs" / "market" / "bbo-1m" / "nbbo" / f"ticker={ticker}"),
            source_rev,
        )

    for symbol in derived["required_futures"]["first_rate_symbol"].tolist():
        run_import(
            source_repo,
            f"data/raw/futures/minute_bars/{symbol}_full_1min_continuous_ratio_adjusted.txt",
            str(imports_root / "data" / "raw" / "futures" / "minute_bars" / f"{symbol}_full_1min_continuous_ratio_adjusted.txt"),
            source_rev,
        )

    for currency in derived["required_fx"]["currency"].tolist():
        run_import(
            source_repo,
            f"data/raw/currencies/minute_bars/{currency}USD_full_1min.txt",
            str(imports_root / "data" / "raw" / "currencies" / "minute_bars" / f"{currency}USD_full_1min.txt"),
            source_rev,
        )


if __name__ == "__main__":
    main()
