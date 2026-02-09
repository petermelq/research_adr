"""
Compute baseline comparison for test stocks.

Compares:
1. Baseline model: predicted_return = beta * index_return
2. LASSO model: predicted_return = beta * index_return + LASSO(russell_residuals)

Both predictions are compared against raw ordinary returns (not residuals).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from utils_lasso_residuals import (
    load_ordinary_exchange_mapping,
    load_index_mapping,
    compute_aligned_returns
)

# Paths
__script_dir__ = Path(__file__).parent.absolute()
data_dir = __script_dir__ / '..' / 'data'
test_model_dir = data_dir / 'processed' / 'models' / 'with_us_stocks' / 'lasso_test'
test_features_dir = data_dir / 'processed' / 'models' / 'with_us_stocks' / 'features_test'

# Test stocks
TEST_STOCKS = ['BP', 'SAP', 'ASML']

def load_model(model_file):
    """Load a trained LASSO model."""
    with open(model_file, 'rb') as f:
        return pickle.load(f)

def main():
    print("=" * 70)
    print("Computing Baseline Comparison")
    print("=" * 70)

    # Load mappings
    print("\nLoading mappings...")
    ordinary_to_exchange, ordinary_to_adr = load_ordinary_exchange_mapping()
    ordinary_to_index, exchange_to_index = load_index_mapping()

    # Create reverse mapping from ADR to ordinary
    adr_to_ordinary = {v: k for k, v in ordinary_to_adr.items()}

    # Load data
    print("Loading data...")
    ordinary_prices = pd.read_csv(data_dir / 'raw' / 'ordinary' / 'ord_PX_LAST_adjust_all.csv', index_col=0, parse_dates=True)
    index_prices = pd.read_csv(data_dir / 'raw' / 'indices' / 'indices_PX_LAST.csv', index_col=0, parse_dates=True)
    betas_df = pd.read_csv(data_dir / 'processed' / 'models' / 'ordinary_betas_index_only.csv', index_col=0, parse_dates=True)

    # Load training metadata
    metadata = pd.read_csv(test_model_dir / 'test_training_metadata.csv')
    metadata['model_date'] = pd.to_datetime(metadata['model_date'])

    print(f"Loaded {len(metadata)} models for {metadata['ticker'].nunique()} stocks")

    # Results list
    results = []

    # Process each model
    for _, row in metadata.iterrows():
        ticker = row['ticker']
        model_date = row['model_date']
        model_file = Path(row['model_file'])

        print(f"\nProcessing {ticker} {model_date.strftime('%Y-%m')}...")

        # Get mappings (ticker in metadata is ADR ticker)
        adr_ticker = ticker
        ordinary_ticker = adr_to_ordinary.get(adr_ticker)

        if ordinary_ticker is None:
            print(f"  ⚠ Warning: No ordinary ticker for ADR {adr_ticker}, skipping")
            continue

        index_symbol = ordinary_to_index.get(ordinary_ticker)

        if index_symbol is None:
            print(f"  ⚠ Warning: No index mapping for {ordinary_ticker}, skipping")
            continue

        # Load model (returns a dict with 'model' key containing the LASSOResidualModel)
        model_dict = load_model(model_file)
        model = model_dict['model']

        # Get test period from saved model data
        test_start, test_end = model_dict['test_period']

        # Load features for this ticker (features are saved with ADR ticker name)
        features_df = pd.read_parquet(test_features_dir / f'{adr_ticker}.parquet')
        features_df.index = pd.to_datetime(features_df.index)

        # Filter to test period (truly out-of-sample)
        test_features = features_df[(features_df.index >= test_start) & (features_df.index <= test_end)]

        if len(test_features) == 0:
            print(f"  ⚠ Warning: No features in test period, skipping")
            continue

        # Get ordinary returns and index returns for test period
        ordinary_returns = compute_aligned_returns(ordinary_prices[[ordinary_ticker]], test_features.index)
        ordinary_returns = ordinary_returns[ordinary_ticker]

        index_returns = compute_aligned_returns(index_prices[[index_symbol]], test_features.index)
        index_returns = index_returns[index_symbol]

        # Get betas
        if adr_ticker in betas_df.columns:
            betas = betas_df.loc[test_features.index, adr_ticker]
        else:
            print(f"  ⚠ Warning: {adr_ticker} not in betas, using beta=1.0")
            betas = pd.Series(1.0, index=test_features.index)

        # Align all data
        common_dates = test_features.index.intersection(ordinary_returns.index).intersection(index_returns.index).intersection(betas.index)

        if len(common_dates) == 0:
            print(f"  ⚠ Warning: No common dates, skipping")
            continue

        # Get aligned data
        y_actual = ordinary_returns.loc[common_dates].values
        index_ret = index_returns.loc[common_dates].values
        beta_vals = betas.loc[common_dates].values

        # Get LASSO residual predictions
        X_features = test_features.loc[common_dates]
        X_residual_cols = [c for c in X_features.columns if c.startswith('russell_')]
        X_residual = X_features[X_residual_cols].fillna(0).values

        # Scale features using model's scaler
        X_scaled = model.scaler.transform(X_residual)
        lasso_residual_pred = model.model.predict(X_scaled)

        # Compute predictions
        # Baseline: predicted = beta * index_return
        baseline_pred = beta_vals * index_ret

        # LASSO: predicted = beta * index_return + LASSO(russell_residuals)
        lasso_full_pred = beta_vals * index_ret + lasso_residual_pred

        # Remove any NaNs
        valid_mask = ~(np.isnan(y_actual) | np.isnan(baseline_pred) | np.isnan(lasso_full_pred))
        y_actual_clean = y_actual[valid_mask]
        baseline_pred_clean = baseline_pred[valid_mask]
        lasso_full_pred_clean = lasso_full_pred[valid_mask]

        if len(y_actual_clean) < 2:
            print(f"  ⚠ Warning: Not enough valid samples, skipping")
            continue

        # Compute ICs
        baseline_ic = np.corrcoef(baseline_pred_clean, y_actual_clean)[0, 1]
        lasso_ic = np.corrcoef(lasso_full_pred_clean, y_actual_clean)[0, 1]

        # Handle NaN correlations (happens when std=0)
        if np.isnan(baseline_ic):
            baseline_ic = 0.0
        if np.isnan(lasso_ic):
            lasso_ic = 0.0

        ic_improvement = lasso_ic - baseline_ic

        print(f"  Baseline IC: {baseline_ic:.4f}")
        print(f"  LASSO IC: {lasso_ic:.4f}")
        print(f"  Improvement: {ic_improvement:+.4f}")
        print(f"  Samples: {len(y_actual_clean)}")

        results.append({
            'ticker': ticker,
            'month': model_date.strftime('%Y-%m'),
            'baseline_ic': baseline_ic,
            'lasso_ic': lasso_ic,
            'ic_improvement': ic_improvement,
            'n_samples': len(y_actual_clean)
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_file = test_model_dir / 'baseline_comparison_corrected.csv'
    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print(f"Saved {len(results_df)} results to {output_file}")
    print("=" * 70)

    # Print summary
    print("\nSUMMARY:")
    print(f"  Mean Baseline IC: {results_df['baseline_ic'].mean():.4f}")
    print(f"  Mean LASSO IC: {results_df['lasso_ic'].mean():.4f}")
    print(f"  Mean Improvement: {results_df['ic_improvement'].mean():.4f}")
    print(f"  % Better: {(results_df['ic_improvement'] > 0).mean() * 100:.1f}%")

    print("\nPer-stock:")
    for ticker in TEST_STOCKS:
        ticker_data = results_df[results_df['ticker'] == ticker]
        if len(ticker_data) > 0:
            print(f"  {ticker}: Baseline={ticker_data['baseline_ic'].mean():.3f}, "
                  f"LASSO={ticker_data['lasso_ic'].mean():.3f}, "
                  f"Improvement={ticker_data['ic_improvement'].mean():+.3f}")

if __name__ == '__main__':
    main()
