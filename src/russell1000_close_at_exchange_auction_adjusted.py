"""
Create adjusted Russell 1000 close prices at foreign exchange auction times.

Reads unadjusted prices from russell1000/close_at_exchange_auction/ and applies
split/dividend adjustments to create adjusted price series.
"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from utils import get_market_business_days, load_params

__script_dir__ = os.path.dirname(os.path.abspath(__file__))


def get_daily_adj(adj_group, start_date, end_date):
    """
    Convert adjustment factors to daily cumulative adjustment series.

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

    # Group by adjustment_date and multiply adjustment factors (handles multiple events same day).
    # This is required so multiple corporate actions on the same ex-date are all applied.
    by_ex_date = adj_group.groupby('adjustment_date')[['adjustment_factor']].prod()

    # Shift index back one business day (ex-date -> last cum-dividend date).
    cbday = get_market_business_days('NYSE')
    by_ex_date.index = [pd.to_datetime(idx) - cbday for idx in by_ex_date.index]

    # Distinct ex-dates can map to the same shifted date around holidays/weekends.
    # Combine those factors on the shifted date before cumprod.
    by_shifted_date = by_ex_date.groupby(level=0)[['adjustment_factor']].prod()

    # Compute cumulative product from most recent backwards.
    by_shifted_date = by_shifted_date.sort_index(ascending=False)
    by_shifted_date['cum_adj'] = by_shifted_date['adjustment_factor'].cumprod()
    cum_adj = by_shifted_date[['cum_adj']].copy()

    # Most recent prices have no adjustment.
    end_ts = pd.Timestamp(end_date)
    cum_adj.loc[end_ts, 'cum_adj'] = 1.0
    cum_adj = cum_adj[~cum_adj.index.duplicated(keep='last')]

    # Create continuous daily series over [start_date, end_date].
    full_index = pd.date_range(pd.Timestamp(start_date), end_ts, freq='1D')
    cum_adj = cum_adj.sort_index().reindex(full_index).bfill().ffill()
    cum_adj.index.name = 'date'

    return cum_adj


def build_daily_adjustment_table(adj_factors, start_date, end_date):
    """
    Build daily cumulative adjustment factors for all tickers.

    Returns:
        DataFrame with columns: ticker, date, cum_adj
    """
    if len(adj_factors) == 0:
        return pd.DataFrame(columns=['ticker', 'date', 'cum_adj'])

    daily_adj = (
        adj_factors.groupby('ticker')
        .apply(get_daily_adj, start_date=start_date, end_date=end_date, include_groups=False)
        .reset_index()
        .rename(columns={'level_1': 'date'})
    )
    return daily_adj


def apply_adjustments(unadj_prices, daily_adj):
    """
    Apply adjustment factors to unadjusted prices.

    Args:
        unadj_prices: DataFrame with dates as index, tickers as columns
        daily_adj: DataFrame with columns: ticker, date, cum_adj

    Returns:
        DataFrame with adjusted prices (same structure as input)
    """
    # If no adjustment factors, return unadjusted prices
    if len(daily_adj) == 0:
        print("    No adjustment factors - returning unadjusted prices")
        return unadj_prices.copy()

    # Stack prices for merging
    stacked_price = unadj_prices.stack().reset_index(name='price')
    stacked_price.columns = ['date', 'ticker', 'price']

    # Merge prices with adjustments
    merged = stacked_price.merge(daily_adj, on=['ticker', 'date'], how='left')

    # Fill missing cum_adj with 1.0 (no adjustment)
    merged['cum_adj'] = merged['cum_adj'].fillna(1.0)

    # Apply adjustments
    merged['adj_price'] = merged['price'] * merged['cum_adj']

    # Pivot back to original structure
    adj_result = merged.pivot(index='date', columns='ticker', values='adj_price')

    return adj_result


def main():
    """Apply adjustments to all exchange close price files."""
    print("=" * 70)
    print("Russell 1000 Adjusted Close Prices at Exchange Auction Times")
    print("=" * 70)

    # Load parameters
    params = load_params()
    start_date = params['frd_start_date']
    end_date = params['end_date']

    print(f"\nDate range: {start_date} to {end_date}")

    # Load adjustment factors
    adj_factors_path = os.path.join(
        __script_dir__,
        '..',
        'data',
        'processed',
        'russell1000',
        'adjustment_factors.csv'
    )

    print(f"\nLoading adjustment factors from: {adj_factors_path}")
    adj_factors = pd.read_csv(adj_factors_path)

    # Remove ' US Equity' suffix from tickers to match price file format
    adj_factors['ticker'] = adj_factors['ticker'].str.replace(' US Equity', '')

    print(f"Loaded adjustment factors for {adj_factors['ticker'].nunique()} tickers")
    print(f"Total adjustment records: {len(adj_factors)}")

    # Input and output directories
    input_dir = Path(__script_dir__) / '..' / 'data' / 'processed' / 'russell1000' / 'close_at_exchange_auction'
    output_dir = Path(__script_dir__) / '..' / 'data' / 'processed' / 'russell1000' / 'close_at_exchange_auction_adjusted'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all CSV files from input directory
    csv_files = sorted(input_dir.glob('*.csv'))

    if not csv_files:
        print(f"\nERROR: No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"\nProcessing {len(csv_files)} exchange files...")

    # Build daily adjustment table once (shared across exchanges)
    print("\nBuilding daily adjustment table...")
    daily_adj = build_daily_adjustment_table(adj_factors, start_date, end_date)
    print(f"Built daily adjustments: {len(daily_adj):,} rows")

    # Process each exchange file
    for csv_file in csv_files:
        exchange_mic = csv_file.stem  # e.g., 'XLON'

        print(f"\n  Processing {exchange_mic}...")

        # Read unadjusted prices
        unadj_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        print(f"    Unadjusted: {unadj_prices.shape[0]} rows x {unadj_prices.shape[1]} columns")

        # Apply adjustments
        adj_prices = apply_adjustments(unadj_prices, daily_adj)

        # Ensure index is sorted
        adj_prices = adj_prices.sort_index()

        print(f"    Adjusted:   {adj_prices.shape[0]} rows x {adj_prices.shape[1]} columns")

        # Save adjusted prices
        output_file = output_dir / f'{exchange_mic}.csv'
        adj_prices.to_csv(output_file)

        print(f"    Saved to: {output_file}")

        # Calculate and display adjustment statistics
        # Count how many tickers had adjustments applied
        adj_tickers = adj_factors['ticker'].unique()
        price_tickers = set(unadj_prices.columns)
        tickers_with_adjustments = len(set(adj_tickers) & price_tickers)

        print(f"    Tickers with adjustments: {tickers_with_adjustments} / {len(price_tickers)}")

    print("\n" + "=" * 70)
    print("Adjustment complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
