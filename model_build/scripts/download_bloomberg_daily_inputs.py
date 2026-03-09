from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run_download(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Bloomberg daily inputs required by model_build.")
    parser.add_argument("--context-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    context_dir = Path(args.context_dir)
    output_root = Path(args.output_root)
    adr_info_path = context_dir / "metadata" / "adr_info.csv"
    required_futures_path = context_dir / "required_futures.csv"
    market_params = context_dir / "market_data_params.yaml"

    params = yaml.safe_load(market_params.read_text(encoding="utf-8"))

    start_date = str(params["start_date"])
    end_date = str(params["end_date"])
    pad_lookback = str(max(params["pred"]["lookback_days"], params["hedge_ratio_lookback_days"]))

    output_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    commands = [
        [
            python,
            "-m",
            "data_tools.cli.download_bbg_daily",
            str(adr_info_path),
            str(output_root / "adrs" / "adr_PX_LAST_adjust_none.csv"),
            "--tickers_columns",
            "adr",
            "--field",
            "PX_LAST",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
            "--overwrite",
        ],
        [
            python,
            "-m",
            "data_tools.cli.download_bbg_daily",
            str(adr_info_path),
            str(output_root / "adrs" / "adr_PX_LAST_adjust_all.csv"),
            "--tickers_columns",
            "adr",
            "--adjust",
            "all",
            "--pad_lookback",
            pad_lookback,
            "--field",
            "PX_LAST",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
            "--overwrite",
        ],
        [
            python,
            "-m",
            "data_tools.cli.download_bbg_daily",
            str(adr_info_path),
            str(output_root / "ordinary" / "ord_PX_LAST_adjust_none.csv"),
            "--tickers_columns",
            "id",
            "--include_suffix",
            "--field",
            "PX_LAST",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
            "--overwrite",
        ],
        [
            python,
            "-m",
            "data_tools.cli.download_bbg_daily",
            str(adr_info_path),
            str(output_root / "ordinary" / "ord_PX_LAST_adjust_split.csv"),
            "--tickers_columns",
            "id",
            "--include_suffix",
            "--adjust",
            "split",
            "--pad_lookback",
            pad_lookback,
            "--field",
            "PX_LAST",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
            "--overwrite",
        ],
        [
            python,
            "-m",
            "data_tools.cli.download_bbg_daily",
            str(adr_info_path),
            str(output_root / "etfs" / "market" / "market_etf_PX_LAST_adjust_none.csv"),
            "--tickers_columns",
            "market_etf_hedge",
            "--symbol_suffix",
            " US Equity",
            "--field",
            "PX_LAST",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
            "--overwrite",
        ],
        [
            python,
            "-m",
            "data_tools.cli.download_bbg_daily",
            str(adr_info_path),
            str(output_root / "etfs" / "market" / "market_etf_PX_LAST_adjust_all.csv"),
            "--tickers_columns",
            "market_etf_hedge",
            "--symbol_suffix",
            " US Equity",
            "--adjust",
            "all",
            "--pad_lookback",
            pad_lookback,
            "--field",
            "PX_LAST",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
            "--overwrite",
        ],
        [
            python,
            "-m",
            "data_tools.cli.download_bbg_daily",
            str(required_futures_path),
            str(output_root / "indices" / "indices_PX_LAST.csv"),
            "--tickers_columns",
            "index",
            "--symbol_suffix",
            " Index",
            "--pad_lookback",
            pad_lookback,
            "--field",
            "PX_LAST",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
            "--overwrite",
        ],
    ]

    for command in commands:
        run_download(command)


if __name__ == "__main__":
    main()
