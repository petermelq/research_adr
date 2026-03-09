from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from adr_strategy_kernel.common import ensure_parent_dir, load_adr_info, load_params, resolve_repo_path
from adr_strategy_kernel.residual_utils import (
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
    is_usd_currency,
    load_fx_minute,
    load_index_currency_mapping,
    load_index_reference_mappings,
    normalize_currency,
)


def compute_index_only_betas(
    output_path: str | Path | None = None,
    index_prices_path: str | Path | None = None,
    ordinary_prices_path: str | Path | None = None,
    adr_info_path: str | Path | None = None,
    close_offsets_path: str | Path | None = None,
    futures_symbols_path: str | Path | None = None,
    fx_dir: str | Path | None = None,
    params_path: str | Path | None = None,
    lookback_days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    output_file = Path(output_path) if output_path is not None else resolve_repo_path(
        "data", "processed", "models", "ordinary_betas_index_only.csv", repo_root=repo_root
    )
    index_file = Path(index_prices_path) if index_prices_path is not None else resolve_repo_path(
        "data", "processed", "aligned_index_prices.csv", repo_root=repo_root
    )
    ordinary_file = Path(ordinary_prices_path) if ordinary_prices_path is not None else resolve_repo_path(
        "data", "raw", "ordinary", "ord_PX_LAST_adjust_split.csv", repo_root=repo_root
    )
    close_offsets_file = Path(close_offsets_path) if close_offsets_path is not None else resolve_repo_path(
        "data", "raw", "close_time_offsets.csv", repo_root=repo_root
    )

    adr_info = load_adr_info(adr_info_path=adr_info_path, repo_root=repo_root)
    adr_info = adr_info.dropna(subset=["adr"])
    adr_info = adr_info[~adr_info["id"].str.contains(" US Equity")]
    adr_info["currency"] = adr_info["currency"].map(normalize_currency)
    index_future_to_symbol, _ = load_index_reference_mappings(
        futures_symbols_path=futures_symbols_path,
        repo_root=repo_root,
    )
    index_to_currency = load_index_currency_mapping(
        futures_symbols_path=futures_symbols_path,
        repo_root=repo_root,
    )
    adr_info["index_symbol"] = adr_info["index_future_bbg"].map(index_future_to_symbol)
    adr_info["index_currency"] = adr_info["index_symbol"].map(index_to_currency)
    tickers = adr_info["id"].tolist()
    adr_dict = adr_info.set_index("id")["adr_ticker"].to_dict()

    params = load_params(params_path=params_path, repo_root=repo_root)
    lookback_days = lookback_days if lookback_days is not None else params["pred"]["lookback_days"]
    start_date = start_date or params["start_date"]
    end_date = end_date or params["end_date"]

    index_data = pd.read_csv(index_file, index_col=0, parse_dates=True)
    ordinary_data = pd.read_csv(ordinary_file, index_col=0, parse_dates=True)
    available_tickers = [ticker for ticker in tickers if ticker in ordinary_data.columns]
    missing_tickers = sorted(set(tickers) - set(available_tickers))
    if missing_tickers:
        print(
            f"Warning: {len(missing_tickers)} tickers missing from {ordinary_file.name}; "
            "excluding from index-only beta fit."
        )
        print(", ".join(missing_tickers))
    if not available_tickers:
        raise RuntimeError(f"No requested tickers available in {ordinary_file}")
    ordinary_data = ordinary_data[available_tickers]

    offsets_df = pd.read_csv(close_offsets_file)
    exchange_offsets = dict(zip(offsets_df["exchange_mic"], offsets_df["offset"]))
    start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    fx_minute_cache: dict[str, pd.Series] = {}
    close_times_cache: dict[str, pd.Series] = {}
    fx_daily_cache: dict[tuple[str, str], pd.Series] = {}

    def get_fx_daily(exchange_mic: str, currency: str) -> pd.Series | None:
        if is_usd_currency(currency):
            return None
        key = (exchange_mic, currency)
        if key in fx_daily_cache:
            return fx_daily_cache[key]
        if currency not in fx_minute_cache:
            fx_minute_cache[currency] = load_fx_minute(currency, fx_dir=fx_dir, repo_root=repo_root)
        if exchange_mic not in close_times_cache:
            offset_str = exchange_offsets.get(exchange_mic, "0min")
            close_times_cache[exchange_mic] = compute_exchange_close_times(
                exchange_mic,
                offset_str,
                start_date_str,
                end_date_str,
            )
        fx_daily_cache[key] = compute_fx_daily_at_close(
            fx_minute_cache[currency],
            close_times_cache[exchange_mic],
        )
        return fx_daily_cache[key]

    index_returns = index_data.pct_change()
    index_returns.columns = [f"{column}_index" for column in index_returns.columns]
    underlying_returns = ordinary_data.pct_change()
    aligned_data = pd.concat([index_returns, underlying_returns], axis=1, join="inner")

    beta_rows: list[tuple[pd.Timestamp, str, float]] = []
    for ordinary_ticker in underlying_returns.columns:
        idx_col = f"{ordinary_ticker}_index"
        if idx_col not in aligned_data.columns:
            print(f"Warning: No aligned index for {ordinary_ticker}, skipping...")
            continue

        valid_data = aligned_data[[idx_col, ordinary_ticker]].dropna()
        if len(valid_data) < 2:
            continue

        ticker_meta = adr_info[adr_info["id"] == ordinary_ticker].iloc[0]
        exchange_mic = ticker_meta["exchange"]
        stock_currency = ticker_meta["currency"]
        index_currency = ticker_meta["index_currency"]
        mismatch_currency = (
            isinstance(stock_currency, str)
            and isinstance(index_currency, str)
            and stock_currency != index_currency
        )

        if mismatch_currency:
            stock_fx = get_fx_daily(exchange_mic, stock_currency)
            index_fx = get_fx_daily(exchange_mic, index_currency)
            converted = pd.DataFrame(index=valid_data.index)
            converted[ordinary_ticker] = (
                convert_returns_to_usd(valid_data[ordinary_ticker], stock_fx)
                if stock_fx is not None
                else valid_data[ordinary_ticker]
            )
            converted[idx_col] = (
                convert_returns_to_usd(valid_data[idx_col], index_fx)
                if index_fx is not None
                else valid_data[idx_col]
            )
            valid_data = converted.dropna()

        start_ts = pd.to_datetime(start_date_str)
        start_loc = int(valid_data.index.searchsorted(start_ts, side="left"))
        start_loc = max(start_loc, lookback_days + 1)

        for row_idx in range(start_loc, len(valid_data)):
            window_start = (
                valid_data.index[row_idx] - pd.Timedelta(days=lookback_days + 1)
            ).strftime("%Y-%m-%d")
            window_end = (
                valid_data.index[row_idx] - pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
            window_data = valid_data.loc[window_start:window_end]
            if len(window_data) < 2:
                continue
            model = LinearRegression()
            model.fit(window_data[[idx_col]], window_data[ordinary_ticker])
            adr_ticker = adr_dict[ordinary_ticker]
            beta_rows.append((valid_data.index[row_idx], adr_ticker, model.coef_[0]))

        print(f"Processed {adr_dict[ordinary_ticker]} ({max(len(valid_data) - start_loc, 0)} beta rows)")

    betas = pd.DataFrame(beta_rows, columns=["date", "ticker", "market_beta"])
    if not betas.empty:
        betas = betas.pivot(index="date", columns="ticker", values="market_beta")
        betas = betas.sort_index().sort_index(axis=1)
    else:
        betas = pd.DataFrame()

    ensure_parent_dir(output_file)
    betas.to_csv(output_file)
    print(f"Betas saved to {output_file}")
    return betas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute rolling index-only ADR betas.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--index-prices", default=None)
    parser.add_argument("--ordinary-prices", default=None)
    parser.add_argument("--adr-info", default=None)
    parser.add_argument("--close-offsets", default=None)
    parser.add_argument("--futures-symbols", default=None)
    parser.add_argument("--fx-dir", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args(argv)

    compute_index_only_betas(
        output_path=args.output,
        index_prices_path=args.index_prices,
        ordinary_prices_path=args.ordinary_prices,
        adr_info_path=args.adr_info,
        close_offsets_path=args.close_offsets,
        futures_symbols_path=args.futures_symbols,
        fx_dir=args.fx_dir,
        params_path=args.params,
        lookback_days=args.lookback_days,
        start_date=args.start_date,
        end_date=args.end_date,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
