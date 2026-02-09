"""
Test the Ridge pipeline on a few stocks.

Runs feature preparation and model training on 3 test stocks to verify
the pipeline works before running on all stocks.
"""

import pandas as pd
import sys
import os
from pathlib import Path

__script_dir__ = Path(__file__).parent.absolute()

# Import the main scripts
sys.path.append(str(__script_dir__))
from prepare_lasso_features import (
    load_data, prepare_features_for_ticker
)
from train_ridge_models import train_rolling_models
from utils_lasso_residuals import (
    load_ordinary_exchange_mapping,
    load_index_mapping,
)


# Test with these stocks (diverse exchanges)
TEST_TICKERS = ['BP', 'SAP', 'ASML']  # UK, Germany, Netherlands


def main():
    """Run pipeline on test stocks."""
    print("=" * 70)
    print("Ridge Pipeline Test - Running on Test Stocks")
    print("=" * 70)

    print(f"\nTest stocks: {TEST_TICKERS}")

    # Set date range
    start_date = '2024-01-02'
    end_date = '2026-01-30'

    print(f"Date range: {start_date} to {end_date}")

    # Load mappings
    print("\nLoading mappings...")
    ordinary_to_exchange, ordinary_to_adr = load_ordinary_exchange_mapping()
    ordinary_to_index, exchange_to_index = load_index_mapping()

    # Reverse mapping to get ordinary tickers from ADR tickers
    adr_to_ordinary = {v: k for k, v in ordinary_to_adr.items()}

    print(f"Total ordinary stocks available: {len(ordinary_to_exchange)}")

    # Load data
    print("\nLoading data files...")
    ordinary_prices, russell_prices_by_exchange, index_prices, betas = load_data()

    print(f"  Ordinary prices: {ordinary_prices.shape}")
    print(f"  Russell exchanges: {list(russell_prices_by_exchange.keys())}")
    print(f"  Index prices: {index_prices.shape}")
    print(f"  Betas: {betas.shape}")

    # Create output directories
    features_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'features_test'
    models_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'ridge_test'

    features_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTest output directories:")
    print(f"  Features: {features_dir}")
    print(f"  Models: {models_dir}")

    # Process each test stock
    print("\n" + "=" * 70)
    print("STEP 1: Preparing Features")
    print("=" * 70)

    success_count = 0
    test_features = {}

    for adr_ticker in TEST_TICKERS:
        print(f"\n[{success_count+1}/{len(TEST_TICKERS)}] Processing {adr_ticker}")

        # Get ordinary ticker
        if adr_ticker not in adr_to_ordinary:
            print(f"  Warning: {adr_ticker} not found in mapping, skipping")
            continue

        ordinary_ticker = adr_to_ordinary[adr_ticker]
        exchange_mic = ordinary_to_exchange.get(ordinary_ticker)
        index_symbol = ordinary_to_index.get(ordinary_ticker)

        if exchange_mic is None or index_symbol is None:
            print(f"  Warning: Missing exchange or index mapping, skipping")
            continue

        if exchange_mic not in russell_prices_by_exchange:
            print(f"  Warning: No Russell data for {exchange_mic}, skipping")
            continue

        russell_prices = russell_prices_by_exchange[exchange_mic]

        try:
            # Prepare features
            features = prepare_features_for_ticker(
                ordinary_ticker, exchange_mic, adr_ticker, index_symbol,
                ordinary_prices, russell_prices, index_prices, betas,
                start_date, end_date
            )

            if features is not None:
                # Save features
                output_file = features_dir / f'{adr_ticker}.parquet'
                features.to_parquet(output_file)
                print(f"  ✓ Saved features to {output_file.name}")

                test_features[adr_ticker] = features
                success_count += 1
            else:
                print(f"  ✗ Feature preparation returned None")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Feature preparation complete: {success_count}/{len(TEST_TICKERS)} stocks")
    print(f"{'=' * 70}")

    if success_count == 0:
        print("\nNo stocks processed successfully. Stopping.")
        return

    # Train models
    print("\n" + "=" * 70)
    print("STEP 2: Training Models")
    print("=" * 70)

    all_metadata = []

    for i, (ticker, features) in enumerate(test_features.items()):
        print(f"\n[{i+1}/{len(test_features)}] Training models for {ticker}")

        try:
            metadata = train_rolling_models(ticker, features, models_dir, train_months=11)
            all_metadata.extend(metadata)

            if len(metadata) > 0:
                print(f"  ✓ Trained {len(metadata)} models")
                print(f"  Average Test IC: {sum(m['test_ic'] for m in metadata) / len(metadata):.4f}")
            else:
                print(f"  ✗ No models trained")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save metadata
    if len(all_metadata) > 0:
        metadata_df = pd.DataFrame(all_metadata)
        metadata_file = models_dir / 'test_training_metadata.csv'
        metadata_df.to_csv(metadata_file, index=False)

        print(f"\n{'=' * 70}")
        print(f"Test complete!")
        print(f"{'=' * 70}")
        print(f"\nResults:")
        print(f"  Stocks processed: {success_count}")
        print(f"  Total models trained: {len(all_metadata)}")
        print(f"  Metadata saved to: {metadata_file}")

        # Summary statistics
        print(f"\nPerformance Summary (Out-of-Sample Test IC):")
        print(f"  Mean Test IC: {metadata_df['test_ic'].mean():.4f}")
        print(f"  Median Test IC: {metadata_df['test_ic'].median():.4f}")
        print(f"  % Positive Test IC: {(metadata_df['test_ic'] > 0).mean() * 100:.1f}%")
        print(f"  Avg non-zero coefs: {metadata_df['n_nonzero_coefs'].mean():.1f}")
        print(f"  Avg alpha: {metadata_df['alpha'].mean():.6f}")

        # Per-stock summary
        print(f"\nPer-Stock Summary:")
        for ticker in metadata_df['ticker'].unique():
            ticker_data = metadata_df[metadata_df['ticker'] == ticker]
            print(f"  {ticker}: {len(ticker_data)} models, avg test IC = {ticker_data['test_ic'].mean():.4f}")

        print(f"\n{'=' * 70}")
        print("Next steps:")
        print("  1. Review results above")
        print("  2. Check model files in:", models_dir)
        print("  3. If results look good, run full pipeline:")
        print("     python src/run_lasso_pipeline.py")
        print("=" * 70)
    else:
        print("\n✗ No models were trained successfully.")


if __name__ == '__main__':
    main()
