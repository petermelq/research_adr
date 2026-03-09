from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adr_strategy_kernel.common import ensure_parent_dir, load_adr_info, load_params, resolve_repo_path
from adr_strategy_kernel.pipelines.fixed_time_mid import load_ticker_list
from adr_strategy_kernel.risk import compute_backtest_covariance


def compute_current_covariance(
    trade_date: str,
    adr_signal_path: str | Path,
    adr_trade_price_path: str | Path,
    adr_close_path: str | Path,
    etf_trade_price_path: str | Path,
    etf_close_path: str | Path,
    hedge_ratios_path: str | Path,
    adr_info_path: str | Path | None = None,
    output_path: str | Path | None = None,
    residuals_output_path: str | Path | None = None,
    metadata_output_path: str | Path | None = None,
    trade_symbols: list[str] | None = None,
    trade_symbols_csv: str | Path | None = None,
    trade_symbols_columns: list[str] | None = None,
    vol_lookback: int = 100,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    trade_symbols_list = None
    if trade_symbols or trade_symbols_csv is not None:
        trade_symbols_list = load_ticker_list(
            tickers=trade_symbols,
            tickers_csv=trade_symbols_csv,
            tickers_columns=trade_symbols_columns,
        )

    adr_info = load_adr_info(adr_info_path=adr_info_path, repo_root=repo_root)
    hedge_map = adr_info.set_index("adr_ticker")["market_etf_hedge"].to_dict()

    covariance, residuals = compute_backtest_covariance(
        trade_date=trade_date,
        adr_signal=pd.read_csv(adr_signal_path, index_col=0, parse_dates=True),
        adr_trade_price=pd.read_csv(adr_trade_price_path, index_col=0, parse_dates=True),
        adr_close=pd.read_csv(adr_close_path, index_col=0, parse_dates=True),
        etf_trade_price=pd.read_csv(etf_trade_price_path, index_col=0, parse_dates=True),
        etf_close=pd.read_csv(etf_close_path, index_col=0, parse_dates=True),
        hedge_ratios=pd.read_csv(hedge_ratios_path, index_col=0, parse_dates=True),
        hedge_map=hedge_map,
        vol_lookback=vol_lookback,
        trade_symbols=trade_symbols_list,
    )

    if output_path is not None:
        output_file = Path(output_path)
        ensure_parent_dir(output_file)
        covariance.to_csv(output_file)
    if residuals_output_path is not None:
        residuals_file = Path(residuals_output_path)
        ensure_parent_dir(residuals_file)
        residuals.to_csv(residuals_file)
    if metadata_output_path is not None:
        metadata_file = Path(metadata_output_path)
        ensure_parent_dir(metadata_file)
        metadata = {
            "trade_date": str(pd.Timestamp(trade_date).date()),
            "vol_lookback": vol_lookback,
            "tickers": covariance.columns.tolist(),
        }
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return covariance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute the trade-date covariance matrix used by hedged_single_time_ADR.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--adr-signal", default=None)
    parser.add_argument("--adr-trade-price", default=None)
    parser.add_argument("--adr-close", default=None)
    parser.add_argument("--etf-trade-price", default=None)
    parser.add_argument("--etf-close", default=None)
    parser.add_argument("--hedge-ratios", default=None)
    parser.add_argument("--adr-info", default=None)
    parser.add_argument("--trade-symbols-csv", default=None)
    parser.add_argument("--trade-symbols-columns", nargs="+", default=None)
    parser.add_argument("--trade-symbols", nargs="*", default=None)
    parser.add_argument("--vol-lookback", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--residuals-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    args = parser.parse_args(argv)
    params = load_params(params_path=args.params, repo_root=args.repo_root)
    trade_date = args.trade_date or params.get("trade_date")
    if trade_date is None:
        raise ValueError("trade_date must be supplied either directly or via --params")
    vol_lookback = args.vol_lookback if args.vol_lookback is not None else params.get("covariance_lookback_days", 100)

    compute_current_covariance(
        trade_date=trade_date,
        adr_signal_path=args.adr_signal or resolve_repo_path("data", "processed", "fixed_time_signal.csv", repo_root=args.repo_root),
        adr_trade_price_path=args.adr_trade_price or resolve_repo_path("data", "processed", "adrs", "adr_daily_fixed_time_mid.csv", repo_root=args.repo_root),
        adr_close_path=args.adr_close or resolve_repo_path("data", "raw", "adrs", "adr_PX_LAST_adjust_none.csv", repo_root=args.repo_root),
        etf_trade_price_path=args.etf_trade_price or resolve_repo_path("data", "processed", "etfs", "market", "market_etf_daily_fixed_time_mid.csv", repo_root=args.repo_root),
        etf_close_path=args.etf_close or resolve_repo_path("data", "raw", "etfs", "market", "market_etf_PX_LAST_adjust_none.csv", repo_root=args.repo_root),
        hedge_ratios_path=args.hedge_ratios or resolve_repo_path("data", "processed", "market_etf_hedge_ratios.csv", repo_root=args.repo_root),
        adr_info_path=args.adr_info,
        output_path=args.output,
        residuals_output_path=args.residuals_output,
        metadata_output_path=args.metadata_output,
        trade_symbols=args.trade_symbols,
        trade_symbols_csv=args.trade_symbols_csv,
        trade_symbols_columns=args.trade_symbols_columns,
        vol_lookback=vol_lookback,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
