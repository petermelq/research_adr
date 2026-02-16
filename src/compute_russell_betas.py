"""
Compute rolling 60-day betas for each Russell 1000 stock vs the corresponding index.

For each exchange, loads adjusted Russell close prices and index prices,
computes daily returns, and calculates rolling betas using covariance/variance.

Outputs one parquet per exchange to data/processed/russell1000/russell_betas/{EXCHANGE}.parquet
Shape: (dates x tickers), values are betas.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from utils_lasso_residuals import (
    load_index_mapping,
    INDEX_TO_FX_CURRENCY,
    load_fx_minute,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)

__script_dir__ = Path(__file__).parent.absolute()

WINDOW = 60
MIN_OBS = 20


def compute_rolling_betas(russell_returns, index_returns, window=WINDOW, min_obs=MIN_OBS):
    """
    Compute rolling betas for all Russell stocks vs index.

    Args:
        russell_returns: DataFrame (dates x tickers) of Russell daily returns
        index_returns: Series of index daily returns
        window: Rolling window size
        min_obs: Minimum observations required

    Returns:
        DataFrame (dates x tickers) of rolling betas
    """
    common_dates = russell_returns.index.intersection(index_returns.index)
    russell_aligned = russell_returns.loc[common_dates]
    index_aligned = index_returns.loc[common_dates]

    betas = pd.DataFrame(index=common_dates, columns=russell_aligned.columns, dtype=float)

    index_vals = index_aligned.values
    russell_vals = russell_aligned.values
    n_dates = len(common_dates)
    n_tickers = russell_vals.shape[1]

    for i in range(window, n_dates):
        idx_window = index_vals[i - window:i]
        russ_window = russell_vals[i - window:i, :]

        for j in range(n_tickers):
            russ_col = russ_window[:, j]
            valid = ~(np.isnan(russ_col) | np.isnan(idx_window))
            if valid.sum() < min_obs:
                continue
            russ_clean = russ_col[valid]
            idx_clean = idx_window[valid]
            var = np.var(idx_clean)
            if var > 0:
                cov = np.cov(russ_clean, idx_clean)[0, 1]
                betas.iloc[i, j] = cov / var

    return betas


def main():
    print("=" * 70)
    print("Computing Russell 1000 Rolling Betas vs Index")
    print("=" * 70)

    params = load_params()
    start_date = params['frd_start_date']
    end_date = params['end_date']
    print(f"Date range: {start_date} to {end_date}")

    # Load mappings
    _, exchange_to_index = load_index_mapping()
    print(f"Exchange-to-index mapping: {exchange_to_index}")

    # Load index prices
    index_path = __script_dir__ / '..' / 'data' / 'raw' / 'indices' / 'indices_PX_LAST.csv'
    index_prices = pd.read_csv(index_path, index_col=0, parse_dates=True)

    # Input/output dirs
    input_dir = __script_dir__ / '..' / 'data' / 'processed' / 'russell1000' / 'close_at_exchange_auction_adjusted'
    output_dir = __script_dir__ / '..' / 'data' / 'processed' / 'russell1000' / 'russell_betas'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load close time offsets and precompute FX daily returns
    offsets_df = pd.read_csv(__script_dir__ / '..' / 'data' / 'raw' / 'close_time_offsets.csv')
    exchange_offsets = dict(zip(offsets_df['exchange_mic'], offsets_df['offset']))

    fx_minute_cache = {}
    fx_daily_by_exchange = {}
    for exchange_mic, index_symbol in exchange_to_index.items():
        currency = INDEX_TO_FX_CURRENCY.get(index_symbol)
        if currency is None:
            continue
        if currency not in fx_minute_cache:
            fx_minute_cache[currency] = load_fx_minute(currency)
        offset_str = exchange_offsets.get(exchange_mic, '0min')
        close_times = compute_exchange_close_times(exchange_mic, offset_str, start_date, end_date)
        fx_daily_by_exchange[exchange_mic] = compute_fx_daily_at_close(
            fx_minute_cache[currency], close_times
        )
        print(f"  {exchange_mic} ({currency}USD): {fx_daily_by_exchange[exchange_mic].dropna().shape[0]} FX return days")

    csv_files = sorted(input_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} exchange files")

    for csv_file in csv_files:
        exchange_mic = csv_file.stem
        index_symbol = exchange_to_index.get(exchange_mic)

        if index_symbol is None:
            print(f"\n  Skipping {exchange_mic}: no index mapping")
            continue

        print(f"\n  Processing {exchange_mic} (index: {index_symbol})...")

        # Load Russell prices for this exchange
        russell_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print(f"    Russell prices: {russell_prices.shape}")

        # Load index prices
        if index_symbol not in index_prices.columns:
            print(f"    Warning: {index_symbol} not in index prices, skipping")
            continue

        index_px = index_prices[[index_symbol]]

        # Compute returns
        russell_returns = russell_prices.pct_change()
        index_ret = index_px[index_symbol].pct_change()

        # Convert index returns to USD
        fx_daily = fx_daily_by_exchange.get(exchange_mic)
        if fx_daily is not None:
            index_ret = convert_returns_to_usd(index_ret, fx_daily)
            print(f"    Converted index returns to USD")

        # Compute rolling betas
        print(f"    Computing rolling {WINDOW}-day betas...")
        betas = compute_rolling_betas(russell_returns, index_ret)

        # Drop rows that are all NaN (before window fills)
        betas = betas.dropna(how='all')
        print(f"    Betas shape: {betas.shape}")

        # Save
        output_file = output_dir / f'{exchange_mic}.parquet'
        betas.to_parquet(output_file)
        print(f"    Saved to {output_file}")

    print("\n" + "=" * 70)
    print("Russell betas computation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
