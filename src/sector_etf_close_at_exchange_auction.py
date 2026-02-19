"""
Extract sector ETF close prices at foreign exchange auction times.

Outputs one CSV per exchange with dates as rows and sector ETF tickers as columns.
Only ETFs referenced in data/raw/sector_etfs.csv for ADRs on that exchange are processed.
"""

from pathlib import Path
import pandas as pd
import pandas_market_calendars as mcal
from tqdm import tqdm
import yaml


def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['frd_start_date'], params['end_date']


def load_close_time_offsets():
    offsets_df = pd.read_csv('data/raw/close_time_offsets.csv')
    exclude = ['XTKS', 'XASX']
    offsets_df = offsets_df[~offsets_df['exchange_mic'].isin(exclude)]
    return dict(zip(offsets_df['exchange_mic'], offsets_df['offset']))


def compute_exchange_auction_times(exchange_mic, offset_str, start_date, end_date):
    cal = mcal.get_calendar(exchange_mic)
    sched = cal.schedule(start_date=start_date, end_date=end_date)
    close_times_et = sched['market_close'].dt.tz_convert('America/New_York')
    close_times_only = close_times_et.dt.time
    most_common_close = close_times_only.mode()[0]
    is_normal_close = close_times_only == most_common_close
    sched_normal = sched[is_normal_close]
    auction_times = (
        sched_normal['market_close'].dt.tz_convert('America/New_York') + pd.Timedelta(offset_str)
    ).dt.tz_localize(None)
    return auction_times


def filter_to_us_trading_days(auction_times, us_schedule):
    valid_dates = auction_times.index.intersection(us_schedule.index)
    us_close_times = (
        us_schedule.loc[valid_dates, 'market_close']
        .dt.tz_convert('America/New_York')
        .dt.tz_localize(None)
    )
    normal_close_time = pd.Timestamp('1900-01-01 16:00:00').time()
    is_normal_close = us_close_times.dt.time == normal_close_time
    normal_close_dates = us_close_times[is_normal_close].index
    return auction_times.loc[auction_times.index.intersection(normal_close_dates)]


def extract_close_for_ticker(parquet_path, exchange_auction_times):
    df = pd.read_parquet(parquet_path, columns=['Close']).sort_index().reset_index()
    ts_col = 'DateTime' if 'DateTime' in df.columns else df.columns[0]

    results = {}
    for exchange_mic, auction_series in exchange_auction_times.items():
        auction_df = auction_series.to_frame(name='auction_time')
        merged = pd.merge_asof(
            auction_df.sort_values('auction_time'),
            df,
            left_on='auction_time',
            right_on=ts_col,
            direction='backward',
        )
        merged['date'] = merged['auction_time'].dt.normalize()
        merged['match_date'] = pd.to_datetime(merged[ts_col]).dt.normalize()
        merged.loc[merged['date'] != merged['match_date'], 'Close'] = float('nan')
        results[exchange_mic] = merged.set_index(merged['auction_time'].dt.date)['Close']
    return results


def main():
    start_date, end_date = load_params()
    offsets = load_close_time_offsets()

    adr_info = pd.read_csv('data/raw/adr_info.csv')
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    sector_map = pd.read_csv('data/raw/sector_etfs.csv')
    sector_map['adr'] = sector_map['adr'].astype(str).str.strip()
    sector_map['hedge'] = sector_map['hedge'].astype(str).str.strip()
    sector_map = sector_map.replace({'hedge': {'': pd.NA, 'nan': pd.NA}}).dropna(subset=['hedge'])

    merged = sector_map.merge(adr_info[['adr', 'exchange']], on='adr', how='inner')
    exchange_to_etfs = merged.groupby('exchange')['hedge'].unique().to_dict()

    us_cal = mcal.get_calendar('NYSE')
    us_schedule = us_cal.schedule(start_date=start_date, end_date=end_date)
    jimmy_carter_date = pd.Timestamp('2025-01-09')
    if jimmy_carter_date in us_schedule.index:
        us_schedule = us_schedule.drop(jimmy_carter_date)

    exchange_auction_times = {}
    for exchange_mic, offset_str in offsets.items():
        if exchange_mic not in exchange_to_etfs:
            continue
        auction_times = compute_exchange_auction_times(exchange_mic, offset_str, start_date, end_date)
        auction_times = filter_to_us_trading_days(auction_times, us_schedule)
        exchange_auction_times[exchange_mic] = auction_times

    data_dir = Path('data/raw/sector_etfs/ohlcv-1m')
    output_dir = Path('data/processed/sector_etfs/close_at_exchange_auction')
    output_dir.mkdir(parents=True, exist_ok=True)

    exchange_results = {ex: {} for ex in exchange_auction_times}

    all_etfs = sorted({etf for v in exchange_to_etfs.values() for etf in v})
    for etf in tqdm(all_etfs, desc='Processing sector ETFs'):
        parquet_path = data_dir / f'ticker={etf}' / 'data.parquet'
        if not parquet_path.exists():
            continue
        ticker_results = extract_close_for_ticker(parquet_path, exchange_auction_times)
        for ex, series in ticker_results.items():
            if etf in set(exchange_to_etfs.get(ex, [])):
                exchange_results[ex][etf] = series

    for ex, ticker_dict in exchange_results.items():
        if not ticker_dict:
            continue
        df = pd.DataFrame(ticker_dict)
        df.index.name = 'date'
        out = output_dir / f'{ex}.csv'
        df.to_csv(out)
        print(f"{ex}: {df.shape} -> {out}")


if __name__ == '__main__':
    main()
