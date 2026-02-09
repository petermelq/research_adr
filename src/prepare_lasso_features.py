"""
Prepare features for LASSO residual prediction.

For each ordinary stock, this script:
1. Loads ordinary and Russell 1000 prices
2. Computes aligned returns
3. Residualizes Russell returns vs index
4. Computes ordinary residuals from existing beta model
5. Saves prepared features for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
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
)

__script_dir__ = Path(__file__).parent.absolute()


def load_data():
    """
    Load all necessary data files.

    Returns:
        tuple: (ordinary_prices, russell_prices_by_exchange, index_prices, betas)
    """
    # Load ordinary prices
    ordinary_path = __script_dir__ / '..' / 'data' / 'raw' / 'ordinary' / 'ord_PX_LAST_adjust_all.csv'
    ordinary_prices = pd.read_csv(ordinary_path, index_col=0, parse_dates=True)

    # Load Russell prices by exchange
    russell_dir = __script_dir__ / '..' / 'data' / 'processed' / 'russell1000' / 'close_at_exchange_auction_adjusted'
    russell_prices_by_exchange = {}
    for exchange_file in russell_dir.glob('*.csv'):
        exchange_mic = exchange_file.stem  # e.g., 'XLON'
        russell_prices_by_exchange[exchange_mic] = pd.read_csv(
            exchange_file, index_col=0, parse_dates=True
        )

    # Load index prices
    index_path = __script_dir__ / '..' / 'data' / 'raw' / 'indices' / 'indices_PX_LAST.csv'
    index_prices = pd.read_csv(index_path, index_col=0, parse_dates=True)

    # Load existing betas
    betas_path = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'ordinary_betas_index_only.csv'
    betas = pd.read_csv(betas_path, index_col=0, parse_dates=True)

    return ordinary_prices, russell_prices_by_exchange, index_prices, betas


def prepare_features_for_ticker(ordinary_ticker, exchange_mic, adr_ticker, index_symbol,
                                ordinary_prices, russell_prices, index_prices, betas,
                                start_date, end_date):
    """
    Prepare features for one ordinary stock.

    Args:
        ordinary_ticker: Ticker for ordinary stock (e.g., 'BP/ LN Equity')
        exchange_mic: Exchange MIC (e.g., 'XLON')
        adr_ticker: ADR ticker (e.g., 'BP')
        index_symbol: Index symbol (e.g., 'UKX')
        ordinary_prices: DataFrame with all ordinary prices
        russell_prices: DataFrame with Russell prices for this exchange
        index_prices: DataFrame with index prices
        betas: DataFrame with time-varying betas
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        DataFrame with columns: date, ordinary_residual, russell_residual_*
    """
    print(f"\n  Processing {ordinary_ticker} (ADR: {adr_ticker}, Exchange: {exchange_mic}, Index: {index_symbol})")

    # Filter to date range
    date_mask = (ordinary_prices.index >= start_date) & (ordinary_prices.index <= end_date)
    ordinary_prices_filtered = ordinary_prices.loc[date_mask]

    date_mask_russell = (russell_prices.index >= start_date) & (russell_prices.index <= end_date)
    russell_prices_filtered = russell_prices.loc[date_mask_russell]

    date_mask_index = (index_prices.index >= start_date) & (index_prices.index <= end_date)
    index_prices_filtered = index_prices.loc[date_mask_index]

    # Get prices for this ticker
    if ordinary_ticker not in ordinary_prices_filtered.columns:
        print(f"    Warning: {ordinary_ticker} not found in ordinary prices, skipping")
        return None

    ordinary_px = ordinary_prices_filtered[[ordinary_ticker]]

    # Get index prices
    if index_symbol not in index_prices_filtered.columns:
        print(f"    Warning: {index_symbol} not found in index prices, skipping")
        return None

    index_px = index_prices_filtered[[index_symbol]]

    # Compute ordinary returns
    # Use all available dates in ordinary prices
    ordinary_returns = compute_aligned_returns(ordinary_px)
    ordinary_returns = ordinary_returns[ordinary_ticker]  # Convert to Series

    # Compute index returns on same dates
    index_returns = compute_aligned_returns(index_px, dates=ordinary_returns.index)
    index_returns = index_returns[index_symbol]  # Convert to Series

    # Compute Russell returns
    # Align to ordinary trading dates
    russell_returns = compute_aligned_returns(russell_prices_filtered, dates=ordinary_returns.index)

    print(f"    Ordinary returns: {len(ordinary_returns.dropna())} non-null values")
    print(f"    Russell returns: {russell_returns.shape[1]} tickers, {russell_returns.notna().sum(axis=1).mean():.0f} avg non-null per date")

    # Residualize Russell returns vs index
    print(f"    Residualizing Russell returns vs {index_symbol}...")
    russell_residuals = residualize_returns(russell_returns, index_returns, window=60)

    # Compute ordinary residuals using existing beta model
    print(f"    Computing ordinary residuals from beta model...")
    ordinary_residuals = get_existing_beta_residuals(
        ordinary_ticker, adr_ticker, ordinary_returns, index_returns, betas
    )

    # Align all data to common dates
    common_dates = (
        ordinary_residuals.index
        .intersection(russell_residuals.index)
    )

    if len(common_dates) == 0:
        print(f"    Warning: No common dates, skipping")
        return None

    print(f"    Common dates: {len(common_dates)}")

    # Create feature dataframe
    features = pd.DataFrame(index=common_dates)
    features['ordinary_residual'] = ordinary_residuals.loc[common_dates]

    # Add Russell residuals as features (use concat to avoid fragmentation)
    russell_residuals_aligned = russell_residuals.loc[common_dates]
    russell_features = russell_residuals_aligned.copy()
    russell_features.columns = [f'russell_{col}' for col in russell_features.columns]
    features = pd.concat([features, russell_features], axis=1)

    # Fill missing values with 0
    features = fill_missing_values(features, fill_value=0.0)

    print(f"    Final features: {features.shape[0]} dates x {features.shape[1]} columns")

    return features


def main():
    """
    Prepare features for all ordinary stocks.
    """
    print("=" * 70)
    print("Preparing LASSO Features")
    print("=" * 70)

    # Set date range
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
    ordinary_prices, russell_prices_by_exchange, index_prices, betas = load_data()

    print(f"  Ordinary prices: {ordinary_prices.shape}")
    print(f"  Russell exchanges: {list(russell_prices_by_exchange.keys())}")
    print(f"  Index prices: {index_prices.shape}")
    print(f"  Betas: {betas.shape}")

    # Create output directory
    output_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Process each ordinary stock
    print("\nProcessing ordinary stocks...")
    success_count = 0
    skip_count = 0

    for ordinary_ticker, exchange_mic in ordinary_to_exchange.items():
        # Get corresponding ADR ticker and index
        adr_ticker = ordinary_to_adr.get(ordinary_ticker)
        index_symbol = ordinary_to_index.get(ordinary_ticker)

        if adr_ticker is None or index_symbol is None:
            print(f"\nSkipping {ordinary_ticker}: missing ADR or index mapping")
            skip_count += 1
            continue

        # Get Russell prices for this exchange
        if exchange_mic not in russell_prices_by_exchange:
            print(f"\nSkipping {ordinary_ticker}: no Russell data for {exchange_mic}")
            skip_count += 1
            continue

        russell_prices = russell_prices_by_exchange[exchange_mic]

        # Prepare features
        try:
            features = prepare_features_for_ticker(
                ordinary_ticker, exchange_mic, adr_ticker, index_symbol,
                ordinary_prices, russell_prices, index_prices, betas,
                start_date, end_date
            )

            if features is not None:
                # Save features
                # Use ADR ticker for filename (cleaner than full ordinary ticker)
                output_file = output_dir / f'{adr_ticker}.parquet'
                features.to_parquet(output_file)
                print(f"    Saved to {output_file.name}")
                success_count += 1
            else:
                skip_count += 1

        except Exception as e:
            print(f"    Error processing {ordinary_ticker}: {e}")
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
