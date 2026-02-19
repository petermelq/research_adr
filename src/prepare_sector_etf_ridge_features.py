"""
Prepare sector-ETF-based features for Ridge residual prediction.

Target remains ordinary_residual (same as existing pipeline).
Feature set is a single residualized sector ETF feature per ticker:
  russell_sector_hedge
"""

from pathlib import Path
from collections import defaultdict
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from utils_lasso_residuals import (
    load_ordinary_exchange_mapping,
    load_index_mapping,
    compute_aligned_returns,
    residualize_returns,
    get_existing_beta_residuals,
    fill_missing_values,
    INDEX_TO_FX_CURRENCY,
    load_fx_minute,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)

__script_dir__ = Path(__file__).parent.absolute()


def main():
    params = load_params()
    start_date = params['frd_start_date']
    end_date = params['end_date']

    ordinary_to_exchange, ordinary_to_adr = load_ordinary_exchange_mapping()
    ordinary_to_index, exchange_to_index = load_index_mapping()

    ordinary_prices = pd.read_csv(
        __script_dir__ / '..' / 'data' / 'raw' / 'ordinary' / 'ord_PX_LAST_adjust_all.csv',
        index_col=0,
        parse_dates=True,
    )
    aligned_index_prices = pd.read_csv(
        __script_dir__ / '..' / 'data' / 'processed' / 'aligned_index_prices.csv',
        index_col=0,
        parse_dates=True,
    )
    betas = pd.read_csv(
        __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'ordinary_betas_index_only.csv',
        index_col=0,
        parse_dates=True,
    )

    sector_dir = __script_dir__ / '..' / 'data' / 'processed' / 'sector_etfs' / 'close_at_exchange_auction'
    sector_prices_by_exchange = {
        p.stem: pd.read_csv(p, index_col=0, parse_dates=True)
        for p in sector_dir.glob('*.csv')
    }

    sector_map = pd.read_csv(__script_dir__ / '..' / 'data' / 'raw' / 'sector_etfs.csv')
    sector_map['adr'] = sector_map['adr'].astype(str).str.strip()
    sector_map['hedge'] = sector_map['hedge'].astype(str).str.strip()
    sector_map = sector_map.replace({'hedge': {'': pd.NA, 'nan': pd.NA}}).dropna(subset=['hedge'])
    adr_to_sector = sector_map.set_index('adr')['hedge'].to_dict()

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

    exchange_to_tickers = defaultdict(list)
    for ordinary_ticker, exchange_mic in ordinary_to_exchange.items():
        adr_ticker = ordinary_to_adr.get(ordinary_ticker)
        if adr_ticker is None:
            continue
        index_symbol = ordinary_to_index.get(ordinary_ticker)
        hedge = adr_to_sector.get(adr_ticker)
        if index_symbol is None or hedge is None:
            continue
        if exchange_mic not in sector_prices_by_exchange:
            continue
        if hedge not in sector_prices_by_exchange[exchange_mic].columns:
            continue
        exchange_to_tickers[exchange_mic].append((ordinary_ticker, adr_ticker, hedge))

    # cache residualized sector features by (exchange, hedge)
    sector_resid_cache = {}
    for exchange_mic, ticker_list in exchange_to_tickers.items():
        rep_ticker = ticker_list[0][0]
        if rep_ticker not in aligned_index_prices.columns:
            continue

        idx_px = aligned_index_prices[[rep_ticker]].dropna()
        idx_px = idx_px.loc[(idx_px.index >= start_date) & (idx_px.index <= end_date)]
        index_returns = compute_aligned_returns(idx_px)
        index_returns = index_returns[rep_ticker]
        fx_daily = fx_daily_by_exchange.get(exchange_mic)
        if fx_daily is not None:
            index_returns = convert_returns_to_usd(index_returns, fx_daily)

        sector_prices = sector_prices_by_exchange[exchange_mic]
        for hedge in sorted({h for _, _, h in ticker_list}):
            s_px = sector_prices[[hedge]].dropna()
            s_px = s_px.loc[(s_px.index >= start_date) & (s_px.index <= end_date)]
            s_ret = compute_aligned_returns(s_px)
            s_ret = s_ret[hedge]
            resid = residualize_returns(s_ret.to_frame(hedge), index_returns, window=60)[hedge]
            sector_resid_cache[(exchange_mic, hedge)] = resid

    output_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'features_sector_etf'
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    for exchange_mic, ticker_list in exchange_to_tickers.items():
        fx_daily = fx_daily_by_exchange.get(exchange_mic)
        for ordinary_ticker, adr_ticker, hedge in ticker_list:
            try:
                ordinary_px = ordinary_prices[[ordinary_ticker]].dropna()
                ordinary_px = ordinary_px.loc[(ordinary_px.index >= start_date) & (ordinary_px.index <= end_date)]
                if ordinary_px.empty:
                    skipped += 1
                    continue

                ordinary_returns = compute_aligned_returns(ordinary_px)[ordinary_ticker]
                if ordinary_ticker not in aligned_index_prices.columns:
                    skipped += 1
                    continue

                idx_px = aligned_index_prices[[ordinary_ticker]].dropna()
                idx_px = idx_px.loc[(idx_px.index >= start_date) & (idx_px.index <= end_date)]
                index_returns = compute_aligned_returns(idx_px, dates=ordinary_returns.index)[ordinary_ticker]

                if fx_daily is not None:
                    index_returns = convert_returns_to_usd(index_returns, fx_daily)
                    ordinary_returns = convert_returns_to_usd(ordinary_returns, fx_daily)

                ordinary_residuals = get_existing_beta_residuals(
                    ordinary_ticker, adr_ticker, ordinary_returns, index_returns, betas
                )

                sector_resid = sector_resid_cache.get((exchange_mic, hedge))
                if sector_resid is None:
                    skipped += 1
                    continue

                common_dates = ordinary_residuals.index.intersection(sector_resid.index)
                if len(common_dates) == 0:
                    skipped += 1
                    continue

                features = pd.DataFrame(index=common_dates)
                features['ordinary_residual'] = ordinary_residuals.loc[common_dates]
                features['russell_sector_hedge'] = sector_resid.loc[common_dates]
                features = fill_missing_values(features, fill_value=0.0)

                out = output_dir / f'{adr_ticker}.parquet'
                features.to_parquet(out)
                success += 1
            except Exception:
                skipped += 1

    print(f"Sector feature prep complete: success={success}, skipped={skipped}")


if __name__ == '__main__':
    main()
