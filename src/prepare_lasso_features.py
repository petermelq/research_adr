"""
Prepare features for LASSO residual prediction.

For each ordinary stock, this script:
1. Loads ordinary and Russell 1000 prices
2. Computes aligned returns
3. Residualizes Russell returns vs index (cached per exchange)
4. Computes ordinary residuals from existing beta model
5. Saves prepared features for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
import sys

sys.path.append(os.path.dirname(__file__))
from utils_lasso_residuals import (
    load_ordinary_exchange_mapping,
    load_index_mapping,
    compute_aligned_returns,
    residualize_returns,
    get_existing_beta_residuals,
    fill_missing_values,
    INDEX_TO_FX_CURRENCY,
    load_fx_minute,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)

__script_dir__ = Path(__file__).parent.absolute()


def main():
    print("=" * 70)
    print("Preparing LASSO Features")
    print("=" * 70)

    start_date = '2024-01-02'
    end_date = '2026-01-30'
    print(f"\nDate range: {start_date} to {end_date}")

    # Load mappings
    print("\nLoading mappings...")
    ordinary_to_exchange, ordinary_to_adr = load_ordinary_exchange_mapping()
    ordinary_to_index, exchange_to_index = load_index_mapping()
    print(f"Found {len(ordinary_to_exchange)} ordinary stocks")

    # Load data
    print("\nLoading data files...")

    ordinary_path = __script_dir__ / '..' / 'data' / 'raw' / 'ordinary' / 'ord_PX_LAST_adjust_all.csv'
    ordinary_prices = pd.read_csv(ordinary_path, index_col=0, parse_dates=True)

    russell_dir = __script_dir__ / '..' / 'data' / 'processed' / 'russell1000' / 'close_at_exchange_auction_adjusted'
    russell_prices_by_exchange = {}
    for exchange_file in russell_dir.glob('*.csv'):
        russell_prices_by_exchange[exchange_file.stem] = pd.read_csv(
            exchange_file, index_col=0, parse_dates=True
        )

    # Aligned index prices: one column per ordinary ticker
    aligned_index_path = __script_dir__ / '..' / 'data' / 'processed' / 'aligned_index_prices.csv'
    aligned_index_prices = pd.read_csv(aligned_index_path, index_col=0, parse_dates=True)

    betas_path = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'ordinary_betas_index_only.csv'
    betas = pd.read_csv(betas_path, index_col=0, parse_dates=True)

    print(f"  Ordinary prices: {ordinary_prices.shape}")
    print(f"  Russell exchanges: {list(russell_prices_by_exchange.keys())}")
    print(f"  Aligned index prices: {aligned_index_prices.shape}")
    print(f"  Betas: {betas.shape}")

    # Precompute FX daily returns per exchange
    print("\nPrecomputing FX daily returns per exchange...")
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

    # Group ordinary tickers by exchange
    exchange_to_tickers = defaultdict(list)
    for ordinary_ticker, exchange_mic in ordinary_to_exchange.items():
        adr_ticker = ordinary_to_adr.get(ordinary_ticker)
        index_symbol = ordinary_to_index.get(ordinary_ticker)
        if adr_ticker is None or index_symbol is None:
            continue
        if exchange_mic not in russell_prices_by_exchange:
            continue
        exchange_to_tickers[exchange_mic].append((ordinary_ticker, adr_ticker, index_symbol))

    # Precompute Russell residuals per exchange (the big speedup)
    print("\nPrecomputing Russell residuals per exchange...")
    russell_residuals_cache = {}

    for exchange_mic, ticker_list in exchange_to_tickers.items():
        print(f"\n  Exchange {exchange_mic} ({len(ticker_list)} tickers)...")

        russell_prices = russell_prices_by_exchange[exchange_mic]
        date_mask = (russell_prices.index >= start_date) & (russell_prices.index <= end_date)
        russell_prices_filtered = russell_prices.loc[date_mask]

        # Get index returns for this exchange from aligned_index_prices
        # All tickers on same exchange share same close time → same index values
        rep_ticker = ticker_list[0][0]  # pick any representative
        if rep_ticker not in aligned_index_prices.columns:
            print(f"    Warning: {rep_ticker} not in aligned_index_prices, skipping exchange")
            continue
        index_px = aligned_index_prices[[rep_ticker]].dropna()
        date_mask_index = (index_px.index >= start_date) & (index_px.index <= end_date)
        index_px = index_px.loc[date_mask_index]

        # Compute Russell returns aligned to Russell trading dates
        russell_returns = compute_aligned_returns(russell_prices_filtered)
        # Compute index returns aligned to Russell dates
        index_returns = compute_aligned_returns(index_px, dates=russell_returns.index)
        index_returns = index_returns[rep_ticker]

        # Convert index returns to USD
        fx_daily = fx_daily_by_exchange.get(exchange_mic)
        if fx_daily is not None:
            index_returns = convert_returns_to_usd(index_returns, fx_daily)

        print(f"    Russell returns: {russell_returns.shape}")
        print(f"    Residualizing Russell returns...")
        russell_residuals = residualize_returns(russell_returns, index_returns, window=60)
        russell_residuals_cache[exchange_mic] = russell_residuals
        print(f"    Russell residuals: {russell_residuals.shape}")

    # Create output directory
    output_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Process each ordinary stock (fast: just ordinary returns + assembly)
    print("\nProcessing ordinary stocks...")
    success_count = 0
    skip_count = 0

    for exchange_mic, ticker_list in exchange_to_tickers.items():
        if exchange_mic not in russell_residuals_cache:
            skip_count += len(ticker_list)
            continue

        russell_residuals = russell_residuals_cache[exchange_mic]
        fx_daily = fx_daily_by_exchange.get(exchange_mic)

        for ordinary_ticker, adr_ticker, index_symbol in ticker_list:
            try:
                # Compute ordinary returns
                date_mask = (ordinary_prices.index >= start_date) & (ordinary_prices.index <= end_date)
                ordinary_px = ordinary_prices.loc[date_mask, [ordinary_ticker]].dropna()

                if ordinary_px.empty:
                    print(f"  Skipping {adr_ticker}: no ordinary prices")
                    skip_count += 1
                    continue

                ordinary_returns = compute_aligned_returns(ordinary_px)
                ordinary_returns = ordinary_returns[ordinary_ticker]

                # Get per-ticker index returns from aligned_index_prices
                if ordinary_ticker not in aligned_index_prices.columns:
                    print(f"  Skipping {adr_ticker}: not in aligned_index_prices")
                    skip_count += 1
                    continue

                idx_px = aligned_index_prices[[ordinary_ticker]].dropna()
                idx_px_filt = idx_px.loc[(idx_px.index >= start_date) & (idx_px.index <= end_date)]
                index_returns = compute_aligned_returns(idx_px_filt, dates=ordinary_returns.index)
                index_returns = index_returns[ordinary_ticker]

                # Convert to USD
                if fx_daily is not None:
                    index_returns = convert_returns_to_usd(index_returns, fx_daily)
                    ordinary_returns = convert_returns_to_usd(ordinary_returns, fx_daily)

                # Compute ordinary residuals from beta model
                ordinary_residuals = get_existing_beta_residuals(
                    ordinary_ticker, adr_ticker, ordinary_returns, index_returns, betas
                )

                # Align to common dates with cached Russell residuals
                common_dates = ordinary_residuals.index.intersection(russell_residuals.index)

                if len(common_dates) == 0:
                    print(f"  Skipping {adr_ticker}: no common dates")
                    skip_count += 1
                    continue

                # Assemble features
                features = pd.DataFrame(index=common_dates)
                features['ordinary_residual'] = ordinary_residuals.loc[common_dates]

                russell_residuals_aligned = russell_residuals.loc[common_dates]
                russell_features = russell_residuals_aligned.copy()
                russell_features.columns = [f'russell_{col}' for col in russell_features.columns]
                features = pd.concat([features, russell_features], axis=1)

                features = fill_missing_values(features, fill_value=0.0)

                output_file = output_dir / f'{adr_ticker}.parquet'
                features.to_parquet(output_file)
                print(f"  {adr_ticker}: {features.shape[0]} dates x {features.shape[1]} cols -> {output_file.name}")
                success_count += 1

            except Exception as e:
                print(f"  Error processing {adr_ticker}: {e}")
                skip_count += 1
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Feature preparation complete!")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped: {skip_count}")
    print("=" * 70)


if __name__ == '__main__':
    main()
