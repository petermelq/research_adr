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
        print("    No adjustment factors - returning unadjusted prices")
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

    # Process each exchange file
    for csv_file in csv_files:
        exchange_mic = csv_file.stem  # e.g., 'XLON'

        print(f"\n  Processing {exchange_mic}...")

        # Read unadjusted prices
        unadj_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        print(f"    Unadjusted: {unadj_prices.shape[0]} rows x {unadj_prices.shape[1]} columns")

        # Apply adjustments
        adj_prices = apply_adjustments(unadj_prices, adj_factors, start_date, end_date)

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
