"""
Download adjustment factors for Russell 1000 stocks from Bloomberg.

This script downloads split and dividend adjustment factors using Bloomberg's
EQY_DVD_ADJUST_FACT field, which provides comprehensive corporate action data.

Note: This script should only be run once to avoid hitting Bloomberg data limits.
The adjustment factors are saved to a CSV file for subsequent processing.
"""

import os
import sys
import pandas as pd
from linux_xbbg import blp

sys.path.append(os.path.dirname(__file__))
from utils import load_params

__script_dir__ = os.path.dirname(os.path.abspath(__file__))


def main():
    """Download adjustment factors from Bloomberg and save to CSV."""
    print("=" * 70)
    print("Russell 1000 Adjustment Factors Download")
    print("=" * 70)

    # Load parameters
    params = load_params()
    start_date = params['frd_start_date']
    end_date = params['end_date']

    print(f"\nDate range: {start_date} to {end_date}")

    # Load Russell 1000 tickers
    tickers_path = os.path.join(__script_dir__, '..', 'data', 'raw', 'russell1000_tickers.csv')
    tickers_df = pd.read_csv(tickers_path)
    tickers = tickers_df['ticker'].tolist()

    print(f"Loaded {len(tickers)} Russell 1000 tickers")

    # Add Bloomberg suffix for US equities
    bbg_tickers = [t + ' US Equity' for t in tickers]

    # Download adjustment factors from Bloomberg BDS
    print(f"\nDownloading adjustment factors from Bloomberg BDS...")
    print("This may take several minutes...")

    try:
        adj_factors = blp.bds(
            bbg_tickers,
            'EQY_DVD_ADJUST_FACT',
            Corporate_Actions_Filter='NORMAL_CASH|ABNORMAL_CASH|CAPITAL_CHANGE',
        )

        if adj_factors is None or len(adj_factors) == 0:
            print("WARNING: No adjustment factors returned from Bloomberg")
            print("Creating empty adjustment factors file")
            adj_factors = pd.DataFrame(columns=[
                'ticker',
                'adjustment_date',
                'adjustment_factor',
                'adjustment_factor_operator_type',
                'adjustment_factor_flag',
            ])
        else:
            # Reset index to make ticker a column
            adj_factors = adj_factors.reset_index(names=['ticker'])

            # Filter to date range
            adj_factors['adjustment_date'] = pd.to_datetime(adj_factors['adjustment_date'])
            adj_factors = adj_factors[
                (adj_factors['adjustment_date'] >= start_date) &
                (adj_factors['adjustment_date'] <= end_date)
            ]

            print(f"\nDownloaded {len(adj_factors)} adjustment factor records")
            print("\nAdjustment Summary:")
            print(f"  Unique tickers: {adj_factors['ticker'].nunique()}")
            if len(adj_factors) > 0:
                print(f"  Date range: {adj_factors['adjustment_date'].min()} to {adj_factors['adjustment_date'].max()}")

                # Show distribution by ticker
                ticker_counts = adj_factors['ticker'].value_counts()
                print(f"\n  Top 10 tickers by number of adjustments:")
                for ticker, count in ticker_counts.head(10).items():
                    print(f"    {ticker}: {count} adjustments")

                # Show breakdown by adjustment type
                print(f"\n  Breakdown by operator type:")
                print(f"    Type 1.0 (divide): {(adj_factors['adjustment_factor_operator_type'] == 1.0).sum()} records")
                print(f"    Type 2.0 (multiply): {(adj_factors['adjustment_factor_operator_type'] == 2.0).sum()} records")

    except Exception as e:
        print(f"ERROR downloading from Bloomberg: {e}")
        print("Creating empty adjustment factors file")
        adj_factors = pd.DataFrame(columns=[
            'ticker',
            'adjustment_date',
            'adjustment_factor',
            'adjustment_factor_operator_type',
            'adjustment_factor_flag',
        ])

    # Save to CSV
    output_path = os.path.join(
        __script_dir__,
        '..',
        'data',
        'processed',
        'russell1000',
        'adjustment_factors.csv'
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    adj_factors.to_csv(output_path, index=False)
    print(f"\nAdjustment factors saved to: {output_path}")
    print("\n" + "=" * 70)
    print("Download complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
