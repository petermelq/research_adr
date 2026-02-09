"""
Unit tests for Russell 1000 adjustment factor application.

Tests the adjustment logic by downloading actual Bloomberg BDS adjustment factors
for NWG, SONY, and GSK, then applying them to ADR unadjusted prices and comparing
against Bloomberg's adjusted prices.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import get_market_business_days
from linux_xbbg import blp


# Test configuration - fixed date range
TEST_START_DATE = '2018-01-01'
TEST_END_DATE = '2026-01-30'
TEST_TICKERS = ['NWG', 'SONY', 'GSK']  # Test subset of ADR tickers


def get_daily_adj(adj_group, start_date, end_date):
    """
    Convert adjustment factors to daily cumulative adjustment series.

    This is the core adjustment logic that will be used in the production code.

    Args:
        adj_group: DataFrame group for a single ticker with adjustment_date, adjustment_factor,
                   and optionally adjustment_factor_operator_type
        start_date: Start date for the series
        end_date: End date for the series

    Returns:
        DataFrame with daily cumulative adjustments indexed by date
    """
    # Convert adjustment factors based on operator type (if present)
    # operator_type == 1.0: divide (invert the factor)
    # operator_type == 2.0: multiply (use as-is)
    # If operator_type not present, assume multiply (type 2.0)
    adj_group = adj_group.copy()
    if 'adjustment_factor_operator_type' in adj_group.columns:
        mask = adj_group['adjustment_factor_operator_type'] == 1.0
        adj_group.loc[mask, 'adjustment_factor'] = 1.0 / adj_group.loc[mask, 'adjustment_factor']

    # Group by adjustment_date and multiply adjustment factors (handles multiple events same day)
    adj_df = adj_group.groupby('adjustment_date')[['adjustment_factor']].prod().sort_index(ascending=False)

    # Set adjustment_factor = 1.0 at start_date (no adjustment for earliest date)
    adj_df.loc[start_date, 'adjustment_factor'] = 1.0

    # Compute cumulative product (creates chain of adjustments from most recent backwards)
    adj_df['cum_adj'] = adj_df['adjustment_factor'].cumprod()

    # Shift index back one business day (ex-date -> last cum-dividend date)
    cbday = get_market_business_days('NYSE')
    adj_df.index = [pd.to_datetime(idx) - cbday for idx in adj_df.index]

    # Set cum_adj = 1.0 at end_date (most recent prices have no adjustment)
    adj_df.loc[end_date, 'cum_adj'] = 1.0

    # Sort and trim to date range
    adj_df = adj_df.sort_index().loc[:end_date]

    # Forward-fill daily to create continuous series
    adj_df = adj_df[['cum_adj']].sort_index().resample('1D').bfill()

    return adj_df


def apply_adjustments(unadj_prices, adj_factors, start_date, end_date):
    """
    Apply adjustment factors to unadjusted prices.

    Args:
        unadj_prices: DataFrame with dates as index, tickers as columns
        adj_factors: DataFrame with columns: ticker, adjustment_date, adjustment_factor
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with adjusted prices (same structure as input)
    """
    # If no adjustment factors, return unadjusted prices
    if len(adj_factors) == 0:
        return unadj_prices.copy()

    # Compute daily cumulative adjustments for each ticker
    adj_df = (adj_factors.groupby('ticker')
                .apply(get_daily_adj, start_date=start_date, end_date=end_date, include_groups=False)
                .reset_index().rename(columns={'level_1': 'date'}))

    # Stack prices for merging
    stacked_price = unadj_prices.stack().reset_index(name='price')
    stacked_price.columns = ['date', 'ticker', 'price']

    # Merge prices with adjustments
    merged = stacked_price.merge(adj_df, on=['ticker', 'date'], how='left')

    # Fill missing cum_adj with 1.0 (no adjustment)
    merged['cum_adj'] = merged['cum_adj'].fillna(1.0)

    # Apply adjustments
    merged['adj_price'] = merged['price'] * merged['cum_adj']

    # Pivot back to original structure
    adj_result = merged.pivot(index='date', columns='ticker', values='adj_price')

    # Reset column names to match expected format (no 'ticker' name attribute)
    adj_result.columns.name = None

    return adj_result


def download_and_cache_adjustment_factors():
    """
    Download adjustment factors from Bloomberg BDS for test tickers.
    Caches results to avoid repeated Bloomberg calls.

    Returns:
        DataFrame with adjustment factors
    """
    cache_file = os.path.join(os.path.dirname(__file__), 'test_adjustment_factors_cache.csv')

    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached adjustment factors from {cache_file}")
        return pd.read_csv(cache_file)

    # Download from Bloomberg
    print(f"Downloading adjustment factors from Bloomberg for {TEST_TICKERS}...")
    bbg_tickers = [f"{ticker} US Equity" for ticker in TEST_TICKERS]

    try:
        adj_factors = blp.bds(
            bbg_tickers,
            'EQY_DVD_ADJUST_FACT',
            Corporate_Actions_Filter='NORMAL_CASH|ABNORMAL_CASH|CAPITAL_CHANGE',
        )

        if adj_factors is None or len(adj_factors) == 0:
            raise ValueError("Bloomberg returned no adjustment factors")

        # Reset index to make ticker a column
        adj_factors = adj_factors.reset_index(names=['ticker'])

        # Filter to test date range
        adj_factors['adjustment_date'] = pd.to_datetime(adj_factors['adjustment_date'])
        adj_factors = adj_factors[
            (adj_factors['adjustment_date'] >= TEST_START_DATE) &
            (adj_factors['adjustment_date'] <= TEST_END_DATE)
        ]

        print(f"Downloaded {len(adj_factors)} adjustment records")
        print(f"Date range: {adj_factors['adjustment_date'].min()} to {adj_factors['adjustment_date'].max()}")

        # Cache results
        adj_factors.to_csv(cache_file, index=False)
        print(f"Cached adjustment factors to {cache_file}")

        return adj_factors

    except Exception as e:
        print(f"Error downloading from Bloomberg: {e}")
        raise


class TestAdjustmentLogic:
    """Test suite for adjustment factor application logic."""

    @pytest.fixture(scope="class")
    def bloomberg_data(self):
        """Download and cache Bloomberg adjustment factors (once per test session)."""
        adj_factors = download_and_cache_adjustment_factors()

        # Load ADR price data
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        unadj_path = os.path.join(base_dir, 'data', 'raw', 'adrs', 'adr_PX_LAST_adjust_none.csv')
        adj_path = os.path.join(base_dir, 'data', 'raw', 'adrs', 'adr_PX_LAST_adjust_all.csv')

        unadj = pd.read_csv(unadj_path, index_col=0, parse_dates=True)
        expected_adj = pd.read_csv(adj_path, index_col=0, parse_dates=True)

        # Filter to test tickers only
        unadj = unadj[TEST_TICKERS]
        expected_adj = expected_adj[TEST_TICKERS]

        # Remove " US Equity" suffix from adjustment factors to match column names
        adj_factors['ticker'] = adj_factors['ticker'].str.replace(' US Equity', '')

        return {
            'adj_factors': adj_factors,
            'unadj': unadj,
            'expected_adj': expected_adj
        }

    def test_bloomberg_adjustment_factors_match(self, bloomberg_data):
        """
        Test that applying Bloomberg BDS adjustment factors to unadjusted ADR prices
        produces the same result as Bloomberg's adjusted prices.

        Tests NWG, SONY, and GSK only.
        """
        adj_factors = bloomberg_data['adj_factors']
        unadj = bloomberg_data['unadj']
        expected_adj = bloomberg_data['expected_adj']

        print(f"\nAdjustment factors summary:")
        print(f"  Tickers: {adj_factors['ticker'].unique()}")
        print(f"  Total adjustments: {len(adj_factors)}")
        for ticker in TEST_TICKERS:
            count = len(adj_factors[adj_factors['ticker'] == ticker])
            print(f"    {ticker}: {count} adjustments")

        # Apply our adjustment logic
        actual_adj = apply_adjustments(unadj, adj_factors, TEST_START_DATE, TEST_END_DATE)

        # Align indices (use intersection of dates present in both)
        common_dates = unadj.index.intersection(expected_adj.index)
        actual_aligned = actual_adj.loc[common_dates, TEST_TICKERS]
        expected_aligned = expected_adj.loc[common_dates, TEST_TICKERS]

        print(f"\nComparing adjusted prices:")
        print(f"  Date range: {common_dates.min()} to {common_dates.max()}")
        print(f"  Number of dates: {len(common_dates)}")

        # Check for differences
        diff = (actual_aligned - expected_aligned).abs()
        max_diff = diff.max().max()
        print(f"  Max absolute difference: {max_diff}")

        # Compare with some tolerance for floating point arithmetic
        pd.testing.assert_frame_equal(
            actual_aligned,
            expected_aligned,
            rtol=1e-4,  # 0.01% relative tolerance (cumulative adjustments accumulate small errors)
            atol=1e-4,  # Small absolute tolerance
            check_dtype=False
        )

        print("✅ Test passed: Adjusted prices match Bloomberg's adjusted prices")

    def test_split_adjustment(self):
        """Test that a 2:1 split (adjustment factor 0.5) correctly adjusts prices."""
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        prices = pd.DataFrame({'TEST': [100.0] * len(dates)}, index=dates)

        adj_factors = pd.DataFrame({
            'ticker': ['TEST'],
            'adjustment_date': [pd.Timestamp('2024-01-06')],
            'adjustment_factor': [0.5]  # 2:1 split
        })

        adj_prices = apply_adjustments(prices, adj_factors, '2024-01-01', '2024-01-10')

        assert abs(adj_prices.loc['2024-01-05', 'TEST'] - 50.0) < 0.01
        assert abs(adj_prices.loc['2024-01-06', 'TEST'] - 100.0) < 0.01

    def test_dividend_adjustment(self):
        """Test that dividend adjustments work correctly."""
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        prices = pd.DataFrame({'TEST': [100.0] * len(dates)}, index=dates)

        adj_factors = pd.DataFrame({
            'ticker': ['TEST'],
            'adjustment_date': [pd.Timestamp('2024-01-06')],
            'adjustment_factor': [0.98]
        })

        adj_prices = apply_adjustments(prices, adj_factors, '2024-01-01', '2024-01-10')

        assert abs(adj_prices.loc['2024-01-05', 'TEST'] - 98.0) < 0.01
        assert abs(adj_prices.loc['2024-01-06', 'TEST'] - 100.0) < 0.01

    def test_cumulative_adjustments(self):
        """Test that multiple adjustments compound correctly."""
        dates = pd.date_range('2024-01-01', '2024-01-15', freq='D')
        prices = pd.DataFrame({'TEST': [100.0] * len(dates)}, index=dates)

        adj_factors = pd.DataFrame({
            'ticker': ['TEST', 'TEST'],
            'adjustment_date': [pd.Timestamp('2024-01-06'), pd.Timestamp('2024-01-11')],
            'adjustment_factor': [0.98, 0.5]
        })

        adj_prices = apply_adjustments(prices, adj_factors, '2024-01-01', '2024-01-15')

        assert abs(adj_prices.loc['2024-01-05', 'TEST'] - 49.0) < 0.01
        assert abs(adj_prices.loc['2024-01-07', 'TEST'] - 50.0) < 0.01
        assert abs(adj_prices.loc['2024-01-15', 'TEST'] - 100.0) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
