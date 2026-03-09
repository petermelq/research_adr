from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from adr_strategy_kernel.common import ensure_parent_dir, load_adr_info, load_params, resolve_repo_path


def compute_market_etf_hedge_ratios(
    adr_prices_path: str | Path | None = None,
    hedge_prices_path: str | Path | None = None,
    adr_info_path: str | Path | None = None,
    output_path: str | Path | None = None,
    params_path: str | Path | None = None,
    lookback_days: int | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    adr_prices_file = Path(adr_prices_path) if adr_prices_path is not None else resolve_repo_path(
        "data", "raw", "adrs", "adr_PX_LAST_adjust_all.csv", repo_root=repo_root
    )
    hedge_prices_file = Path(hedge_prices_path) if hedge_prices_path is not None else resolve_repo_path(
        "data", "raw", "etfs", "market", "market_etf_PX_LAST_adjust_all.csv", repo_root=repo_root
    )
    output_file = Path(output_path) if output_path is not None else resolve_repo_path(
        "data", "processed", "market_etf_hedge_ratios.csv", repo_root=repo_root
    )

    params = load_params(params_path=params_path, repo_root=repo_root)
    lookback_days = lookback_days if lookback_days is not None else params.get(
        "hedge_ratio_lookback_days",
        params["pred"]["lookback_days"],
    )

    adr_info = load_adr_info(adr_info_path=adr_info_path, repo_root=repo_root)
    adr_tickers = adr_info["adr_ticker"].tolist()
    hedge_map = adr_info.set_index("adr_ticker")["market_etf_hedge"].to_dict()

    hedge_data = pd.read_csv(hedge_prices_file, index_col=0, parse_dates=True).sort_index()
    adr_data = pd.read_csv(adr_prices_file, index_col=0, parse_dates=True).sort_index()
    hedge_returns = hedge_data.pct_change()
    adr_returns = adr_data.pct_change()

    beta_dict: dict[tuple[pd.Timestamp, str], float] = {}
    for ticker in adr_tickers:
        hedge_ticker = hedge_map.get(ticker)
        if ticker not in adr_returns.columns or hedge_ticker not in hedge_returns.columns:
            continue

        valid_data = pd.concat([adr_returns[ticker], hedge_returns[hedge_ticker]], axis=1).dropna()
        if len(valid_data) < lookback_days + 1:
            continue

        for row_idx in range(lookback_days, len(valid_data)):
            window_data = valid_data.iloc[row_idx - lookback_days : row_idx]
            model = LinearRegression()
            model.fit(window_data[[hedge_ticker]], window_data[ticker])
            beta_dict[(valid_data.index[row_idx], ticker)] = model.coef_[0]

    result = pd.Series(beta_dict).unstack().sort_index() if beta_dict else pd.DataFrame()
    ensure_parent_dir(output_file)
    result.to_csv(output_file)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute rolling ADR-vs-market-ETF hedge ratios.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--adr-prices", default=None)
    parser.add_argument("--hedge-prices", default=None)
    parser.add_argument("--adr-info", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    args = parser.parse_args(argv)

    compute_market_etf_hedge_ratios(
        adr_prices_path=args.adr_prices,
        hedge_prices_path=args.hedge_prices,
        adr_info_path=args.adr_info,
        output_path=args.output,
        params_path=args.params,
        lookback_days=args.lookback_days,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
