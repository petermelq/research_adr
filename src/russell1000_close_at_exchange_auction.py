"""
Extract Russell 1000 Close prices at foreign exchange closing auction times.

For each foreign exchange (excluding Asia exchanges that close before US extended hours),
extract the Close price from Russell 1000 minute bar data at the exchange's closing auction time.

Outputs one CSV per exchange with dates as row index, tickers as columns.
"""

import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm


def load_params():
    """Load parameters from params.yaml."""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['frd_start_date'], params['end_date']


def load_close_time_offsets():
    """Load close time offsets from CSV, excluding Asia exchanges used in the special branch."""
    offsets_df = pd.read_csv('data/raw/close_time_offsets.csv')
    # Exclude exchanges that close before US extended hours (4 AM ET)
    exclude = ['XTKS', 'XASX', 'XHKG', 'XSES', 'XSHG', 'XSHE']
    offsets_df = offsets_df[~offsets_df['exchange_mic'].isin(exclude)]
    return dict(zip(offsets_df['exchange_mic'], offsets_df['offset']))


def compute_exchange_auction_times(exchange_mic, offset_str, start_date, end_date):
    """
    Get naive ET auction datetimes for a foreign exchange, filtered for normal close days only.

    Args:
        exchange_mic: Exchange MIC code (e.g., 'XLON')
        offset_str: Offset string compatible with pd.Timedelta (e.g., '2 minutes')
        start_date: Start date for schedule
        end_date: End date for schedule

    Returns:
        Series of naive ET datetime auction times indexed by date (only normal close days)
    """
    cal = mcal.get_calendar(exchange_mic)
    sched = cal.schedule(start_date=start_date, end_date=end_date)

    # Determine normal close in the exchange's local timezone.
    # Using ET wall-clock times here is wrong for non-US venues because
    # US and local DST transitions happen on different dates.
    close_times_local = sched['market_close'].dt.tz_convert(str(cal.tz))
    close_times_only_local = close_times_local.dt.time
    most_common_close = close_times_only_local.mode()[0]

    # Filter to only normal close days (exclude early close days)
    is_normal_close = close_times_only_local == most_common_close
    sched_normal = sched[is_normal_close]

    # Convert market close to ET, add offset, then strip timezone
    auction_times = (
        sched_normal['market_close']
        .dt.tz_convert('America/New_York') + pd.Timedelta(offset_str)
    ).dt.tz_localize(None)

    return auction_times


def filter_to_us_trading_days(auction_times, us_schedule):
    """
    Filter auction times to only include days when US market is open
    with normal hours (exclude holidays and early close days).

    Args:
        auction_times: Series of naive ET datetime auction times indexed by date
        us_schedule: DataFrame from pandas_market_calendars with market_open and market_close

    Returns:
        Filtered Series of auction times
    """
    # Get dates of auction times (foreign exchange trading days)
    auction_dates = auction_times.index

    # Get dates when US market is open
    us_open_dates = us_schedule.index

    # Filter to only dates that are both foreign exchange trading days AND US trading days
    valid_dates = auction_dates.intersection(us_open_dates)

    # Convert US market close times to naive ET
    us_close_times = (
        us_schedule.loc[valid_dates, 'market_close']
        .dt.tz_convert('America/New_York')
        .dt.tz_localize(None)
    )

    # Normal US market close is 4:00 PM ET (16:00)
    # Filter out early close days by checking if close time is 4:00 PM
    normal_close_time = pd.Timestamp('1900-01-01 16:00:00').time()
    is_normal_close = us_close_times.dt.time == normal_close_time

    # Keep only dates with normal close times
    normal_close_dates = us_close_times[is_normal_close].index
    auction_times_filtered = auction_times.loc[auction_times.index.intersection(normal_close_dates)]

    return auction_times_filtered


def extract_close_for_ticker(parquet_path, exchange_auction_times):
    """
    For a single ticker, extract Close at each exchange's auction time via merge_asof.

    Args:
        parquet_path: Path to ticker's parquet file
        exchange_auction_times: Dict of {exchange_mic: Series of auction times}

    Returns:
        Dict of {exchange_mic: Series of Close prices indexed by date}
    """
    # Read ticker data
    df = pd.read_parquet(parquet_path, columns=['Close']).sort_index().reset_index()

    results = {}
    for exchange_mic, auction_series in exchange_auction_times.items():
        auction_df = auction_series.to_frame(name='auction_time')

        # Merge asof to find the closest minute bar at or before auction time
        merged = pd.merge_asof(
            auction_df.sort_values('auction_time'),
            df,
            left_on='auction_time',
            right_on='DateTime',
            direction='backward'
        )

        # Guard: NaN out matches from wrong day (e.g., stock halted, no data that day)
        merged['date'] = merged['auction_time'].dt.normalize()
        merged['match_date'] = merged['DateTime'].dt.normalize()
        merged.loc[merged['date'] != merged['match_date'], 'Close'] = float('nan')

        # Return series indexed by date
        results[exchange_mic] = merged.set_index(merged['auction_time'].dt.date)['Close']

    return results


def main():
    """Main execution function."""
    print("Loading parameters...")
    start_date, end_date = load_params()
    print(f"Date range: {start_date} to {end_date}")

    print("\nLoading close time offsets...")
    offsets = load_close_time_offsets()
    print(f"Loaded {len(offsets)} exchanges (excluding Asia branch exchanges)")

    print("\nGetting US market schedule...")
    us_cal = mcal.get_calendar('NYSE')
    us_schedule = us_cal.schedule(start_date=start_date, end_date=end_date)

    # Add Jimmy Carter's funeral as a US market holiday (2025-01-09)
    jimmy_carter_date = pd.Timestamp('2025-01-09')
    if jimmy_carter_date in us_schedule.index:
        us_schedule = us_schedule.drop(jimmy_carter_date)
        print(f"US market trading days: {len(us_schedule)} (added Jimmy Carter funeral holiday on 2025-01-09)")
    else:
        print(f"US market trading days: {len(us_schedule)}")

    print("\nComputing auction times for each exchange (filtered for early closes and US trading days)...")
    exchange_auction_times = {}
    for exchange_mic, offset_str in offsets.items():
        # Get schedule to count total days before any filtering
        cal = mcal.get_calendar(exchange_mic)
        sched = cal.schedule(start_date=start_date, end_date=end_date)
        total_days = len(sched)

        # Get auction times (already filtered for foreign exchange early close days)
        auction_times = compute_exchange_auction_times(
            exchange_mic, offset_str, start_date, end_date
        )

        # Filter to only US trading days (normal hours only, no early close)
        auction_times_filtered = filter_to_us_trading_days(auction_times, us_schedule)
        exchange_auction_times[exchange_mic] = auction_times_filtered

        print(f"  {exchange_mic}: {total_days} total -> {len(auction_times)} after foreign early close filter -> {len(auction_times_filtered)} final")

    print("\nScanning ticker directories...")
    data_dir = Path('data/raw/russell1000/ohlcv-1m')
    ticker_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('ticker=')])
    print(f"Found {len(ticker_dirs)} tickers")

    # Initialize results containers - one per exchange
    exchange_results = {exchange_mic: {} for exchange_mic in offsets.keys()}

    print("\nExtracting Close prices at auction times...")
    for ticker_dir in tqdm(ticker_dirs, desc="Processing tickers"):
        ticker = ticker_dir.name.split('=')[1]
        parquet_path = ticker_dir / 'data.parquet'

        if not parquet_path.exists():
            print(f"Warning: {parquet_path} does not exist, skipping")
            continue

        try:
            ticker_results = extract_close_for_ticker(parquet_path, exchange_auction_times)

            # Add to exchange results
            for exchange_mic, close_series in ticker_results.items():
                exchange_results[exchange_mic][ticker] = close_series
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    print("\nCreating output DataFrames and saving...")
    output_dir = Path('data/processed/russell1000/close_at_exchange_auction')
    output_dir.mkdir(parents=True, exist_ok=True)

    for exchange_mic, ticker_dict in exchange_results.items():
        if not ticker_dict:
            print(f"Warning: No data for {exchange_mic}, skipping")
            continue

        # Create DataFrame with dates as index, tickers as columns
        df = pd.DataFrame(ticker_dict)
        df.index.name = 'date'

        output_path = output_dir / f'{exchange_mic}.csv'
        df.to_csv(output_path)
        print(f"  {exchange_mic}: {df.shape[0]} rows x {df.shape[1]} columns -> {output_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
