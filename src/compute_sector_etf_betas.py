"""
Compute rolling 60-day betas for sector ETFs vs exchange index.

Outputs one parquet per exchange to data/processed/sector_etfs/sector_etf_betas/{EXCHANGE}.parquet
"""

from pathlib import Path
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from utils_lasso_residuals import (
    load_index_mapping,
    load_ordinary_exchange_mapping,
    INDEX_TO_FX_CURRENCY,
    load_fx_minute,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)
from compute_russell_betas import compute_rolling_betas

__script_dir__ = Path(__file__).parent.absolute()


def main():
    params = load_params()
    start_date = params['frd_start_date']
    end_date = params['end_date']

    _, exchange_to_index = load_index_mapping()
    ordinary_to_exchange, _ = load_ordinary_exchange_mapping()

    aligned_index_prices = pd.read_csv(
        __script_dir__ / '..' / 'data' / 'processed' / 'aligned_index_prices.csv',
        index_col=0,
        parse_dates=True,
    )

    exchange_to_rep_ticker = {}
    for ordinary_ticker, exchange_mic in ordinary_to_exchange.items():
        if ordinary_ticker in aligned_index_prices.columns and exchange_mic not in exchange_to_rep_ticker:
            exchange_to_rep_ticker[exchange_mic] = ordinary_ticker

    offsets_df = pd.read_csv(__script_dir__ / '..' / 'data' / 'raw' / 'close_time_offsets.csv')
    exchange_offsets = dict(zip(offsets_df['exchange_mic'], offsets_df['offset']))

    fx_minute_cache = {}
    fx_daily_by_exchange = {}
    for exchange_mic, index_symbol in exchange_to_index.items():
        currency = INDEX_TO_FX_CURRENCY.get(index_symbol)
        if currency is None:
            continue
        if currency not in fx_minute_cache:
            fx_minute_cache[currency] = load_fx_minute(currency)
        offset_str = exchange_offsets.get(exchange_mic, '0min')
        close_times = compute_exchange_close_times(exchange_mic, offset_str, start_date, end_date)
        fx_daily_by_exchange[exchange_mic] = compute_fx_daily_at_close(
            fx_minute_cache[currency], close_times
        )

    input_dir = __script_dir__ / '..' / 'data' / 'processed' / 'sector_etfs' / 'close_at_exchange_auction'
    output_dir = __script_dir__ / '..' / 'data' / 'processed' / 'sector_etfs' / 'sector_etf_betas'
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(input_dir.glob('*.csv')):
        exchange_mic = csv_file.stem
        if exchange_mic not in exchange_to_index:
            continue

        sector_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        if sector_prices.empty:
            continue

        rep_ticker = exchange_to_rep_ticker.get(exchange_mic)
        if rep_ticker is None or rep_ticker not in aligned_index_prices.columns:
            continue

        index_px = aligned_index_prices[rep_ticker].dropna()
        sector_ret = sector_prices.pct_change()
        index_ret = index_px.pct_change()

        fx_daily = fx_daily_by_exchange.get(exchange_mic)
        if fx_daily is not None:
            index_ret = convert_returns_to_usd(index_ret, fx_daily)

        betas = compute_rolling_betas(sector_ret, index_ret)
        betas = betas.dropna(how='all')
        out = output_dir / f'{exchange_mic}.parquet'
        betas.to_parquet(out)
        print(f"{exchange_mic}: {betas.shape} -> {out}")


if __name__ == '__main__':
    main()
