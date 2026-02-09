"""
Unit tests for LASSO residual prediction utilities.
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils_lasso_residuals import (
    compute_aligned_returns,
    residualize_returns,
    fill_missing_values,
)


class TestComputeAlignedReturns:
    """Test compute_aligned_returns function."""

    def test_basic_returns(self):
        """Test basic return calculation."""
        # Create simple price series
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        prices = pd.DataFrame({
            'A': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        }, index=dates)

        returns = compute_aligned_returns(prices)

        # Check returns are approximately 2%
        expected_returns = np.array([np.nan, 0.02, 0.0196, 0.0192, 0.0189,
                                     0.0185, 0.0182, 0.0179, 0.0175, 0.0172])

        actual_returns = returns['A'].values

        # First return is NaN
        assert np.isnan(actual_returns[0])

        # Rest should be close to expected
        np.testing.assert_array_almost_equal(actual_returns[1:], expected_returns[1:], decimal=3)

    def test_with_holiday(self):
        """Test return calculation with missing date (holiday)."""
        # Create price series with gap
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-04', '2024-01-05'])
        prices = pd.DataFrame({
            'A': [100, 102, 106, 108]  # 2024-01-03 is missing
        }, index=dates)

        # Target dates include the holiday
        target_dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        returns = compute_aligned_returns(prices, dates=target_dates)

        # On 2024-01-03, uses most recent two dates (01-01 and 01-02)
        # so return = (102 - 100) / 100 = 0.02
        np.testing.assert_almost_equal(returns.loc['2024-01-03', 'A'], 0.02, decimal=4)

        # Return from 102 to 106 is about 3.92%
        expected_return_01_04 = (106 - 102) / 102
        np.testing.assert_almost_equal(returns.loc['2024-01-04', 'A'], expected_return_01_04, decimal=4)

    def test_with_missing_values(self):
        """Test handling of NaN values in prices."""
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        prices = pd.DataFrame({
            'A': [100, 102, np.nan, 106, 108]
        }, index=dates)

        returns = compute_aligned_returns(prices)

        # Returns should be NaN where prices are NaN
        assert np.isnan(returns.loc['2024-01-03', 'A'])


class TestResidualizeReturns:
    """Test residualize_returns function."""

    def test_perfect_beta_one(self):
        """Test residualization with perfect beta=1 relationship."""
        # Create returns with perfect beta=1
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')

        # Index returns
        index_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

        # Stock returns = index returns (beta = 1)
        stock_returns = pd.DataFrame({
            'A': index_returns.values
        }, index=dates)

        # Residualize
        residuals = residualize_returns(stock_returns, index_returns, window=60)

        # After warmup period, residuals should be near zero
        residuals_after_warmup = residuals.iloc[60:]['A'].dropna()

        # Mean absolute residual should be very small
        assert residuals_after_warmup.abs().mean() < 0.001

    def test_zero_correlation(self):
        """Test residualization with uncorrelated returns."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')

        np.random.seed(42)
        index_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

        # Independent stock returns
        np.random.seed(43)
        stock_returns = pd.DataFrame({
            'A': np.random.randn(len(dates)) * 0.01
        }, index=dates)

        # Residualize
        residuals = residualize_returns(stock_returns, index_returns, window=60)

        # Residuals should be similar to original returns (beta ~ 0)
        residuals_after_warmup = residuals.iloc[60:]['A'].dropna()
        stock_after_warmup = stock_returns.iloc[60:]['A'].dropna()

        # Correlation should be high
        corr = np.corrcoef(residuals_after_warmup, stock_after_warmup)[0, 1]
        assert corr > 0.95

    def test_window_size(self):
        """Test that window size affects the residualization."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')

        index_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        stock_returns = pd.DataFrame({
            'A': index_returns.values + np.random.randn(len(dates)) * 0.001
        }, index=dates)

        # Residualize with different windows
        residuals_30 = residualize_returns(stock_returns, index_returns, window=30)
        residuals_60 = residualize_returns(stock_returns, index_returns, window=60)

        # Should produce different results
        diff = (residuals_30 - residuals_60).dropna()
        assert diff.abs().mean().mean() > 0  # Some difference


class TestFillMissingValues:
    """Test fill_missing_values function."""

    def test_basic_fill(self):
        """Test basic NaN filling."""
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [np.nan, 2.0, np.nan, 4.0, np.nan]
        })

        filled = fill_missing_values(df, fill_value=0.0)

        assert filled['A'].tolist() == [1.0, 0.0, 3.0, 0.0, 5.0]
        assert filled['B'].tolist() == [0.0, 2.0, 0.0, 4.0, 0.0]

    def test_custom_fill_value(self):
        """Test filling with custom value."""
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0]
        })

        filled = fill_missing_values(df, fill_value=-999.0)

        assert filled['A'].tolist() == [1.0, -999.0, 3.0]

    def test_no_missing(self):
        """Test with no missing values."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })

        filled = fill_missing_values(df)

        pd.testing.assert_frame_equal(filled, df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
