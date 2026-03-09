from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import (
    artifact_paths,
    derive_start_dates,
    derive_required_inputs,
    load_artifact_tables,
    load_manifest,
    parse_hhmm,
    write_json,
    write_yaml,
)


def render_model_artifact(artifact_dir: str | Path, output_dir: str | Path) -> None:
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(artifact_dir)
    tables = load_artifact_tables(artifact_dir)
    derived = derive_required_inputs(tables["adr_info"], tables["futures_symbols"])
    start_dates = derive_start_dates(manifest, tables["adr_info"])
    trade_symbols = tables["trade_symbols"] if tables["trade_symbols"] is not None else derived["adr_tickers"]

    for name, source in artifact_paths(artifact_dir).items():
        if name == "manifest":
            continue
        shutil.copy2(source, metadata_dir / source.name)

    trade_symbols_path = output_dir / "trade_symbols.csv"
    trade_symbols.to_csv(trade_symbols_path, index=False)
    derived["adr_tickers"].to_csv(output_dir / "adr_tickers.csv", index=False)
    derived["market_etf_tickers"].to_csv(output_dir / "market_etf_tickers.csv", index=False)
    derived["required_futures"].to_csv(output_dir / "required_futures.csv", index=False)
    derived["required_fx"].to_csv(output_dir / "required_fx.csv", index=False)

    trade_hours, trade_minutes = parse_hhmm(manifest["fixed_trade_time"])
    common_params = {
        "fixed_trade_time_hours": trade_hours,
        "fixed_trade_time_min": trade_minutes,
        "trade_date": str(manifest["trade_date"]),
        "pred": {
            "lookback_days": int(manifest["lookbacks"]["index_beta_days"]),
        },
        "hedge_ratio_lookback_days": int(manifest["lookbacks"]["hedge_ratio_days"]),
        "covariance_lookback_days": int(manifest["lookbacks"]["covariance_days"]),
    }
    market_params = dict(common_params)
    market_params.update(
        {
            "start_date": start_dates["market_data_start_date"],
            "end_date": str(manifest["trade_date"]),
        }
    )
    signal_params = dict(common_params)
    signal_params.update(
        {
            "start_date": start_dates["intraday_start_date"],
            "end_date": str(manifest["trade_date"]),
        }
    )
    write_yaml(output_dir / "market_data_params.yaml", market_params)
    write_yaml(output_dir / "signal_params.yaml", signal_params)

    runtime = {
        "artifact_dir": str(artifact_dir),
        "trade_date": str(manifest["trade_date"]),
        "market_data_start_date": start_dates["market_data_start_date"],
        "intraday_start_date": start_dates["intraday_start_date"],
        "fixed_trade_time": str(manifest["fixed_trade_time"]),
        "lookbacks": manifest["lookbacks"],
        "kernel": manifest["kernel"],
        "sources": manifest["sources"],
        "counts": {
            "adr_tickers": int(len(derived["adr_tickers"])),
            "market_etf_tickers": int(len(derived["market_etf_tickers"])),
            "trade_symbols": int(len(trade_symbols)),
            "required_futures": int(len(derived["required_futures"])),
            "required_fx": int(len(derived["required_fx"])),
        },
    }
    write_json(output_dir / "runtime_config.json", runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a model_build artifact into runtime context files.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    render_model_artifact(args.artifact_dir, args.output_dir)


if __name__ == "__main__":
    main()
