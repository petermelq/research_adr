import os
import pandas as pd
import pandas_market_calendars as mcal
import sys
sys.path.append('../src/')
import utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Load ADR info
    adr_info = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/adr_info.csv'))
    adr_info = adr_info.dropna(subset=['adr'])
    adr_info = adr_info[~adr_info['id'].str.contains(' US Equity')]
    tickers = adr_info['id'].tolist()

    # Load futures symbols
    futures_symbols = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/futures_symbols.csv'))
    futures_to_index = futures_symbols.set_index('bloomberg_symbol')['index'].to_dict()
    bbg_to_frd = futures_symbols.set_index('bloomberg_symbol')['first_rate_symbol'].dropna().to_dict()

    # Mappings
    stock_to_index_future = adr_info.set_index('id')['index_future_bbg'].to_dict()
    stock_to_index = {stock: futures_to_index.get(idx_future)
                      for stock, idx_future in stock_to_index_future.items()}
    stock_to_exchange = adr_info.set_index('id')['exchange'].to_dict()

    # Load raw index prices
    index_data = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/indices/indices_PX_LAST.csv'),
                             index_col=0, parse_dates=True)

    # Load close times
    close_times_df = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/bloomberg_close_times.csv'),
                                 index_col=0)

    # Load close time offsets
    offsets_df = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/close_time_offsets.csv'))
    offsets = dict(zip(offsets_df['exchange_mic'], offsets_df['offset']))

    def parse_minutes_since_midnight(t):
        """Parse time string 'HH:MM:SS' to minutes since midnight."""
        parts = str(t).split(':')
        return int(parts[0]) * 60 + int(parts[1])

    # Classify stocks as aligned or misaligned
    misaligned_stocks = set()
    for col in tickers:
        idx_symbol = stock_to_index.get(col)
        if idx_symbol is None:
            continue
        idx_ticker = f"{idx_symbol} Index"
        if col not in close_times_df.index or idx_ticker not in close_times_df.index:
            continue
        stock_min = parse_minutes_since_midnight(close_times_df.loc[col, 'BLOOMBERG_CLOSE_TIME'])
        index_min = parse_minutes_since_midnight(close_times_df.loc[idx_ticker, 'BLOOMBERG_CLOSE_TIME'])
        if abs(stock_min - index_min) > 10:
            misaligned_stocks.add(col)
            print(f"Misaligned: {col} (close={close_times_df.loc[col, 'BLOOMBERG_CLOSE_TIME']}) "
                  f"vs {idx_ticker} (close={close_times_df.loc[idx_ticker, 'BLOOMBERG_CLOSE_TIME']})")

    print(f"Aligned: {len(tickers) - len(misaligned_stocks)}, "
          f"Misaligned: {len(misaligned_stocks)}")

    # Pre-compute futures-at-close prices per exchange
    data_start = index_data.index.min().strftime('%Y-%m-%d')
    data_end = index_data.index.max().strftime('%Y-%m-%d')

    exchange_futures_at_close = {}
    for col in misaligned_stocks:
        exchange = stock_to_exchange[col]
        if exchange in exchange_futures_at_close:
            continue

        index_future = stock_to_index_future[col]
        frd_symbol = bbg_to_frd.get(index_future)
        if frd_symbol is None:
            print(f"Warning: No FRD symbol for index_future={index_future}")
            continue

        offset = offsets.get(exchange, '0min')
        cal = mcal.get_calendar(exchange)
        sched = cal.schedule(start_date=data_start, end_date=data_end)

        # Filter half-days using LOCAL timezone (constant across DST)
        local_tz = str(cal.tz)
        close_local = sched['market_close'].dt.tz_convert(local_tz)
        normal_local_time = close_local.dt.time.mode()[0]
        is_normal = close_local.dt.time == normal_local_time
        sched = sched[is_normal]

        # Convert to ET and add auction offset
        close_times = sched['market_close'].dt.tz_convert('America/New_York') + pd.Timedelta(offset)

        # Load FRD futures minute bars
        futures_path = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'minute_bars',
                                    f'{frd_symbol}_full_1min_continuous_ratio_adjusted.txt')
        futures_df = pd.read_csv(futures_path, header=None,
                                 names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        futures_df['timestamp'] = futures_df['timestamp'].dt.tz_localize('America/New_York')
        futures_minute = futures_df.set_index('timestamp')['close']

        # Sample futures at exchange close times (searchsorted acts as ffill)
        futures_at_close = pd.Series(index=close_times.index, dtype=float)
        for date, close_time in close_times.items():
            idx = futures_minute.index.searchsorted(close_time, side='right') - 1
            if idx >= 0:
                futures_at_close.loc[date] = futures_minute.iloc[idx]

        futures_at_close = futures_at_close.dropna()

        # Normalize index to timezone-naive dates
        futures_at_close.index = pd.to_datetime(futures_at_close.index.date)

        exchange_futures_at_close[exchange] = futures_at_close
        print(f"Computed futures-at-close for {exchange} using {frd_symbol}: {len(futures_at_close)} days")

    # Build aligned index price DataFrame (one column per ordinary ticker)
    aligned_prices = {}
    for col in tickers:
        idx_symbol = stock_to_index.get(col)
        if idx_symbol is None:
            continue

        exchange = stock_to_exchange.get(col)
        if col in misaligned_stocks and exchange in exchange_futures_at_close:
            aligned_prices[col] = exchange_futures_at_close[exchange]
        else:
            if idx_symbol in index_data.columns:
                aligned_prices[col] = index_data[idx_symbol]

    result = pd.DataFrame(aligned_prices)
    result.index.name = 'date'

    output_filename = os.path.join(SCRIPT_DIR, '../data/processed/aligned_index_prices.csv')
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(output_filename)
    print(f"Aligned index prices saved to {output_filename}")
    print(f"Shape: {result.shape}")
