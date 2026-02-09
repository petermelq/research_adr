"""
Unit tests for LASSO model training.
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_lasso_models import LASSOResidualModel, get_rolling_windows


class TestLASSOResidualModel:
    """Test LASSOResidualModel class."""

    def test_fit_and_predict(self):
        """Test basic fit and predict functionality."""
        # Create synthetic data with known relationships
        np.random.seed(42)
        n_samples = 200
        n_features = 50

        # True coefficients (sparse)
        true_coefs = np.zeros(n_features)
        true_coefs[:5] = [2.0, -1.5, 1.0, -0.5, 0.3]

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate target with noise
        y = X @ true_coefs + np.random.randn(n_samples) * 0.1

        # Convert to DataFrames
        X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
        y_series = pd.Series(y)

        # Split train/val
        split = 150
        X_train, X_val = X_df.iloc[:split], X_df.iloc[split:]
        y_train, y_val = y_series.iloc[:split], y_series.iloc[split:]

        # Fit model
        model = LASSOResidualModel(alpha_grid=np.logspace(-4, 0, 10))
        metrics = model.fit(X_train, y_train, X_val, y_val)

        # Check metrics
        assert 'val_ic' in metrics
        assert 'best_alpha' in metrics
        assert 'n_nonzero_coefs' in metrics

        # Validation IC should be positive (good fit)
        assert metrics['val_ic'] > 0.5

        # Model should be sparse
        assert metrics['n_nonzero_coefs'] < n_features

        # Test prediction
        predictions = model.predict(X_val)
        assert len(predictions) == len(y_val)

        # Predictions should correlate with actual
        corr = np.corrcoef(predictions, y_val)[0, 1]
        assert corr > 0.5

    def test_get_coefficients(self):
        """Test coefficient retrieval."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 2.0 + np.random.randn(n_samples) * 0.1

        X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
        y_series = pd.Series(y)

        # Split
        split = 80
        X_train, X_val = X_df.iloc[:split], X_df.iloc[split:]
        y_train, y_val = y_series.iloc[:split], y_series.iloc[split:]

        # Fit
        model = LASSOResidualModel(alpha_grid=[0.01])
        model.fit(X_train, y_train, X_val, y_val)

        # Get coefficients
        coefs = model.get_coefficients()

        # Should be DataFrame
        assert isinstance(coefs, pd.DataFrame)
        assert 'feature' in coefs.columns
        assert 'coefficient' in coefs.columns

        # Should have all features
        assert len(coefs) == n_features

        # Sorted by absolute value (descending)
        assert coefs['coefficient'].abs().is_monotonic_decreasing

    def test_standardization(self):
        """Test that features are standardized."""
        np.random.seed(42)
        n_samples = 100

        # Features with different scales
        X = pd.DataFrame({
            'feat_0': np.random.randn(n_samples) * 1000,  # Large scale
            'feat_1': np.random.randn(n_samples) * 0.001,  # Small scale
        })
        y = pd.Series(X['feat_0'] / 1000 + X['feat_1'] * 1000 + np.random.randn(n_samples) * 0.1)

        # Split
        split = 80
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        # Fit
        model = LASSOResidualModel(alpha_grid=[0.01])
        metrics = model.fit(X_train, y_train, X_val, y_val)

        # Should still get reasonable IC despite scale differences
        assert metrics['val_ic'] > 0.3


class TestGetRollingWindows:
    """Test get_rolling_windows function."""

    def test_basic_windows(self):
        """Test basic window generation."""
        # Create 24 months of dates (through Feb 2026 to ensure last month has data)
        dates = pd.date_range('2024-01-01', '2026-02-28', freq='D')

        windows = get_rolling_windows(dates, train_months=11)

        # Should have at least 12 windows (25 months - 12 months)
        assert len(windows) >= 12

        # Check structure
        for window in windows:
            assert 'train_start' in window
            assert 'train_end' in window
            assert 'val_start' in window
            assert 'val_end' in window
            assert 'model_date' in window

            # Train should come before val
            assert window['train_start'] < window['train_end']
            assert window['train_end'] < window['val_start']
            assert window['val_start'] <= window['val_end']  # Can be equal if only one day in val month

    def test_window_overlap(self):
        """Test that windows are rolling (overlapping)."""
        dates = pd.date_range('2024-01-01', '2026-01-01', freq='D')

        windows = get_rolling_windows(dates, train_months=11)

        # Check that consecutive windows overlap
        if len(windows) >= 2:
            w1 = windows[0]
            w2 = windows[1]

            # Second window's training should overlap with first
            assert w2['train_start'] > w1['train_start']
            assert w2['train_start'] < w1['train_end']

    def test_insufficient_data(self):
        """Test with insufficient data."""
        # Only 6 months of data
        dates = pd.date_range('2024-01-01', '2024-07-01', freq='D')

        windows = get_rolling_windows(dates, train_months=11)

        # Should have no windows (need 12 months)
        assert len(windows) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
