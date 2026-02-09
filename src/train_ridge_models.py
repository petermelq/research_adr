"""
Train Ridge regression models for ordinary stock residual prediction.

For each ordinary stock:
- Uses rolling 12-month windows
- Refits monthly
- Tunes regularization parameter on validation set
- Saves trained models with metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import sys

__script_dir__ = Path(__file__).parent.absolute()


class RidgeResidualModel:
    """
    Ridge model for predicting ordinary stock residuals.

    Features are standardized before fitting.
    No intercept is included.
    """

    def __init__(self, alpha_grid=None):
        """
        Initialize Ridge model.

        Args:
            alpha_grid: Array of alpha values to search (default: logspace(-4, 1, 50))
        """
        if alpha_grid is None:
            alpha_grid = np.logspace(-4, 1, 50)

        self.alpha_grid = alpha_grid
        self.scaler = StandardScaler()
        self.best_alpha = None
        self.model = None
        self.feature_names = None
        self.train_period = None
        self.val_period = None

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit LASSO with hyperparameter tuning.

        Process:
        1. Standardize features on training set
        2. Grid search over alpha using validation set IC
        3. Retrain on full data (train + val) with best alpha

        Args:
            X_train: Training features (DataFrame)
            y_train: Training targets (Series)
            X_val: Validation features (DataFrame)
            y_val: Validation targets (Series)

        Returns:
            dict: Validation metrics
        """
        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Convert to numpy arrays
        X_train_arr = X_train.values
        y_train_arr = y_train.values
        X_val_arr = X_val.values
        y_val_arr = y_val.values

        # Fit scaler on training data only
        self.scaler.fit(X_train_arr)

        # Transform train and validation
        X_train_scaled = self.scaler.transform(X_train_arr)
        X_val_scaled = self.scaler.transform(X_val_arr)

        # Grid search over alpha
        best_ic = -np.inf
        best_alpha = None

        for alpha in self.alpha_grid:
            # Fit model
            model = Ridge(alpha=alpha, fit_intercept=False, max_iter=10000)
            model.fit(X_train_scaled, y_train_arr)

            # Predict on validation
            y_val_pred = model.predict(X_val_scaled)

            # Compute IC (correlation between predictions and actuals)
            ic = self._compute_ic(y_val_pred, y_val_arr)

            if ic > best_ic:
                best_ic = ic
                best_alpha = alpha

        # Retrain on full data with best alpha
        X_full = np.vstack([X_train_arr, X_val_arr])
        y_full = np.concatenate([y_train_arr, y_val_arr])

        # Refit scaler on full data
        self.scaler.fit(X_full)
        X_full_scaled = self.scaler.transform(X_full)

        # Fit final model
        self.model = Ridge(alpha=best_alpha, fit_intercept=False, max_iter=10000)
        self.model.fit(X_full_scaled, y_full)

        self.best_alpha = best_alpha

        # Return validation metrics
        return {
            'val_ic': best_ic,
            'best_alpha': best_alpha,
            'n_nonzero_coefs': np.sum(self.model.coef_ != 0)
        }

    def predict(self, X):
        """
        Predict using fitted model.

        Args:
            X: Features (DataFrame or numpy array)

        Returns:
            numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")

        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        return self.model.predict(X_scaled)

    def get_coefficients(self):
        """
        Get model coefficients with feature names.

        Returns:
            DataFrame with columns: feature, coefficient
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")

        coefs = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })

        # Sort by absolute value
        coefs['abs_coef'] = coefs['coefficient'].abs()
        coefs = coefs.sort_values('abs_coef', ascending=False)
        coefs = coefs.drop('abs_coef', axis=1)

        return coefs

    def _compute_ic(self, predictions, actuals):
        """
        Compute Information Coefficient.

        IC = correlation between predictions and actuals.

        Returns 0 if not enough valid samples.
        """
        # Remove NaNs
        mask = ~(np.isnan(predictions) | np.isnan(actuals))

        if mask.sum() < 2:
            return 0.0

        predictions_clean = predictions[mask]
        actuals_clean = actuals[mask]

        # Check for zero variance
        if np.std(predictions_clean) == 0 or np.std(actuals_clean) == 0:
            return 0.0

        # Compute correlation
        return np.corrcoef(predictions_clean, actuals_clean)[0, 1]


def get_rolling_windows(dates, train_months=11, refit_freq='M'):
    """
    Generate rolling window date ranges for walk-forward validation.

    For each test month M, creates a window with:
    - Train: M-12 to M-2 (11 months) - for hyperparameter tuning
    - Validate: M-1 (1 month) - for selecting best alpha
    - Retrain: M-12 to M-1 (12 months) - train+val combined, used for final model
    - Test: M (1 month) - truly out-of-sample prediction

    Args:
        dates: DatetimeIndex of available dates
        train_months: Number of months for training (default 11)
        refit_freq: Frequency for refitting ('M' for monthly)

    Returns:
        list of dicts with keys: train_start, train_end, val_start, val_end,
                                  test_start, test_end, model_date
    """
    # Get unique months
    dates = pd.DatetimeIndex(dates).sort_values()
    months = dates.to_period('M').unique().sort_values()

    windows = []

    # Need: 11 train + 1 val + 1 test = 13 months minimum
    total_months_needed = train_months + 2  # 11 train + 1 val + 1 test

    for i in range(total_months_needed, len(months) + 1):
        # Get months for this window
        # Test month is the last month (index i-1)
        test_month = months[i - 1]
        # Val month is second to last (index i-2)
        val_month = months[i - 2]
        # Train months are M-12 to M-2 (11 months before val)
        train_months_range = months[i - total_months_needed:i - 2]

        # Get actual dates
        train_dates = dates[dates.to_period('M').isin(train_months_range)]
        val_dates = dates[dates.to_period('M') == val_month]
        test_dates = dates[dates.to_period('M') == test_month]

        if len(train_dates) == 0 or len(val_dates) == 0 or len(test_dates) == 0:
            continue

        windows.append({
            'train_start': train_dates.min(),
            'train_end': train_dates.max(),
            'val_start': val_dates.min(),
            'val_end': val_dates.max(),
            'test_start': test_dates.min(),
            'test_end': test_dates.max(),
            'model_date': test_dates.max(),  # Model predicts test month
        })

    return windows


def train_rolling_models(ticker, features_df, output_dir, train_months=11):
    """
    Train models on rolling windows.

    Args:
        ticker: Stock ticker (for file naming)
        features_df: DataFrame with date index and columns: ordinary_residual, russell_*
        output_dir: Directory to save models
        train_months: Number of months for training window

    Returns:
        list: Metadata for all trained models
    """
    print(f"\n{'=' * 70}")
    print(f"Training Ridge models for {ticker}")
    print(f"{'=' * 70}")

    # Get target and features
    target_col = 'ordinary_residual'
    feature_cols = [col for col in features_df.columns if col.startswith('russell_')]

    y = features_df[target_col]
    X = features_df[feature_cols]

    print(f"Features: {len(feature_cols)} Russell stocks")
    print(f"Dates: {features_df.index.min()} to {features_df.index.max()} ({len(features_df)} days)")

    # Get rolling windows
    windows = get_rolling_windows(features_df.index, train_months=train_months)

    print(f"Rolling windows: {len(windows)}")

    if len(windows) == 0:
        print("Warning: No windows generated, skipping")
        return []

    # Create output directory for this ticker
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Train models
    models_metadata = []

    for i, window in enumerate(windows):
        print(f"\nWindow {i+1}/{len(windows)}: {window['model_date'].strftime('%Y-%m')}")
        print(f"  Train: {window['train_start'].strftime('%Y-%m-%d')} to {window['train_end'].strftime('%Y-%m-%d')}")
        print(f"  Val:   {window['val_start'].strftime('%Y-%m-%d')} to {window['val_end'].strftime('%Y-%m-%d')}")
        print(f"  Test:  {window['test_start'].strftime('%Y-%m-%d')} to {window['test_end'].strftime('%Y-%m-%d')}")

        # Get train/val/test data
        train_mask = (features_df.index >= window['train_start']) & (features_df.index <= window['train_end'])
        val_mask = (features_df.index >= window['val_start']) & (features_df.index <= window['val_end'])
        test_mask = (features_df.index >= window['test_start']) & (features_df.index <= window['test_end'])

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        print(f"  Train samples: {len(y_train)}, Val samples: {len(y_val)}, Test samples: {len(y_test)}")

        # Skip if not enough data
        if len(y_train) < 20 or len(y_val) < 5 or len(y_test) < 5:
            print(f"  Skipping: insufficient data")
            continue

        # Train model (fits on train, tunes on val, retrains on train+val)
        model = RidgeResidualModel()
        metrics = model.fit(X_train, y_train, X_val, y_val)

        # Evaluate on test set (truly out-of-sample)
        y_test_pred = model.predict(X_test)
        test_ic = model._compute_ic(y_test_pred, y_test.values)

        print(f"  Best alpha: {metrics['best_alpha']:.6f}")
        print(f"  Val IC (for tuning): {metrics['val_ic']:.4f}")
        print(f"  Test IC (out-of-sample): {test_ic:.4f}")
        print(f"  Non-zero coefs: {metrics['n_nonzero_coefs']}")

        # Get top features
        coefs = model.get_coefficients()
        top_features = coefs.head(5)['feature'].tolist()

        # Save model
        model_date_str = window['model_date'].strftime('%Y_%m')
        model_file = ticker_dir / f'{model_date_str}.pkl'

        model_data = {
            'model': model,
            'alpha': metrics['best_alpha'],
            'train_period': (window['train_start'], window['train_end']),
            'val_period': (window['val_start'], window['val_end']),
            'test_period': (window['test_start'], window['test_end']),
            'model_date': window['model_date'],
            'val_ic': metrics['val_ic'],
            'test_ic': test_ic,
            'n_nonzero_coefs': metrics['n_nonzero_coefs'],
            'top_features': top_features,
            'feature_names': feature_cols,
        }

        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"  Saved to {model_file.name}")

        # Store metadata
        models_metadata.append({
            'ticker': ticker,
            'model_date': window['model_date'],
            'alpha': metrics['best_alpha'],
            'val_ic': metrics['val_ic'],
            'test_ic': test_ic,
            'n_nonzero_coefs': metrics['n_nonzero_coefs'],
            'top_5_features': ','.join(top_features),
            'model_file': str(model_file),
        })

    print(f"\n{'=' * 70}")
    print(f"Completed {ticker}: {len(models_metadata)} models trained")
    print(f"{'=' * 70}")

    return models_metadata


def main():
    """
    Train Ridge models for all ordinary stocks.
    """
    print("=" * 70)
    print("LASSO Model Training")
    print("=" * 70)

    # Set paths
    features_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'features'
    models_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'ridge'

    print(f"\nFeatures directory: {features_dir}")
    print(f"Models directory: {models_dir}")

    # Get all feature files
    feature_files = list(features_dir.glob('*.parquet'))
    print(f"\nFound {len(feature_files)} feature files")

    if len(feature_files) == 0:
        print("No feature files found. Run prepare_lasso_features.py first.")
        return

    # Train models for each ticker
    all_metadata = []

    for i, feature_file in enumerate(feature_files):
        ticker = feature_file.stem  # Get ticker from filename

        print(f"\n[{i+1}/{len(feature_files)}] Processing {ticker}")

        try:
            # Load features
            features = pd.read_parquet(feature_file)

            # Train models
            metadata = train_rolling_models(ticker, features, models_dir)

            all_metadata.extend(metadata)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # Save all metadata
    if len(all_metadata) > 0:
        metadata_df = pd.DataFrame(all_metadata)
        metadata_file = models_dir / 'training_metadata.csv'
        metadata_df.to_csv(metadata_file, index=False)

        print(f"\n{'=' * 70}")
        print(f"Training complete!")
        print(f"  Total models trained: {len(all_metadata)}")
        print(f"  Unique tickers: {metadata_df['ticker'].nunique()}")
        print(f"  Metadata saved to: {metadata_file}")
        print(f"{'=' * 70}")
    else:
        print("\nNo models were trained successfully.")


if __name__ == '__main__':
    main()
