"""
Generate sector-ETF Ridge-augmented intraday signal.

Signal = futures_only_signal + ridge_residual_pred,
where ridge_residual_pred is predicted from one sector ETF residual feature.
"""

import gc
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from tqdm import tqdm

from utils import load_params

__script_dir__ = Path(__file__).parent.absolute()

TIME_FUTURES_AFTER_CLOSE = {
    'XLON': pd.Timedelta('6min'), 'XAMS': pd.Timedelta('6min'), 'XPAR': pd.Timedelta('6min'),
    'XETR': pd.Timedelta('6min'), 'XMIL': pd.Timedelta('6min'), 'XBRU': pd.Timedelta('6min'),
    'XMAD': pd.Timedelta('6min'), 'XHEL': pd.Timedelta('0min'), 'XDUB': pd.Timedelta('0min'),
    'XOSL': pd.Timedelta('5min'), 'XSTO': pd.Timedelta('0min'), 'XSWX': pd.Timedelta('1min'),
    'XCSE': pd.Timedelta('0min'), 'XTKS': pd.Timedelta('1min'), 'XASX': pd.Timedelta('11min'),
}


def _precompute_linear_params(model_obj):
    coef = model_obj.model.coef_.astype(np.float32)
    scale = model_obj.scaler.scale_.astype(np.float32)
    mean = model_obj.scaler.mean_.astype(np.float32)
    safe_scale = np.where(scale == 0, 1.0, scale)
    w = coef / safe_scale
    c = -float(np.dot(mean / safe_scale, coef))
    return float(w[0]), c


def load_models(model_dir):
    models = {}
    model_start = None
    model_end = None
    for ticker_dir in sorted(model_dir.glob('*')):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        entries = []
        for mf in sorted(ticker_dir.glob('*.pkl')):
            with open(mf, 'rb') as f:
                md = pickle.load(f)
            test_start, test_end = map(pd.Timestamp, md['test_period'])
            w, c = _precompute_linear_params(md['model'])
            entries.append({'test_start': test_start, 'test_end': test_end, 'w': w, 'c': c})
            model_start = test_start if model_start is None or test_start < model_start else model_start
            model_end = test_end if model_end is None or test_end > model_end else model_end
        if entries:
            models[ticker] = entries
    return models, model_start, model_end


def get_model_for_date(entries, dt):
    for e in entries:
        if e['test_start'] <= dt <= e['test_end']:
            return e
    return None


def main():
    params = load_params()
    start_date, end_date = params['start_date'], params['end_date']

    data_dir = __script_dir__ / '..'
    futures_dir = data_dir / 'data' / 'processed' / 'futures' / 'converted_minute_bars'
    sector_ohlcv_dir = data_dir / 'data' / 'raw' / 'sector_etfs' / 'ohlcv-1m'
    sector_betas_dir = data_dir / 'data' / 'processed' / 'sector_etfs' / 'sector_etf_betas'
    model_dir = data_dir / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'ridge_sector_etf'
    baseline_signal_dir = data_dir / 'data' / 'processed' / 'futures_only_signal'
    output_dir = data_dir / 'data' / 'processed' / 'index_sector_etf_ridge_signal'

    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.glob('ticker=*'):
        shutil.rmtree(p, ignore_errors=True)

    adr_info = pd.read_csv(data_dir / 'data' / 'raw' / 'adr_info.csv')
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    exchange_dict = adr_info.set_index('adr')['exchange'].to_dict()

    sector_map = pd.read_csv(data_dir / 'data' / 'raw' / 'sector_etfs.csv')
    sector_map['adr'] = sector_map['adr'].astype(str).str.strip()
    sector_map['hedge'] = sector_map['hedge'].astype(str).str.strip()
    sector_map = sector_map.replace({'hedge': {'': pd.NA, 'nan': pd.NA}}).dropna(subset=['hedge'])
    adr_to_sector = sector_map.set_index('adr')['hedge'].to_dict()

    futures_symbols = pd.read_csv(data_dir / 'data' / 'raw' / 'futures_symbols.csv')
    adr_info['index_future_bbg'] = adr_info['index_future_bbg'].str.strip()
    futures_symbols['bloomberg_symbol'] = futures_symbols['bloomberg_symbol'].str.strip()
    merged_info = adr_info.merge(futures_symbols, left_on='index_future_bbg', right_on='bloomberg_symbol')
    stock_to_frd = merged_info.set_index('adr')['first_rate_symbol'].to_dict()

    models_by_ticker, model_start, model_end = load_models(model_dir)
    eligible = sorted([t for t in models_by_ticker if t in adr_to_sector and t in exchange_dict])
    print(f"Loaded sector models for {len(models_by_ticker)} tickers; eligible={len(eligible)}")

    eligible_by_exchange = {}
    for t in eligible:
        eligible_by_exchange.setdefault(exchange_dict[t], []).append(t)

    futures_cache = {}
    sector_cache = {}
    augmented_any = set()

    t0 = time.perf_counter()
    for ex, tickers in sorted(eligible_by_exchange.items()):
        offset = TIME_FUTURES_AFTER_CLOSE[ex]
        close_df = (
            mcal.get_calendar(ex).schedule(start_date=start_date, end_date=end_date)['market_close']
            .dt.tz_convert('America/New_York')
        )
        close_df.index = close_df.index.astype(str)

        betas_path = sector_betas_dir / f'{ex}.parquet'
        if not betas_path.exists():
            continue
        sector_betas = pd.read_parquet(betas_path)
        sector_betas.index = pd.to_datetime(sector_betas.index)

        futures_symbol = None
        for t in tickers:
            fs = stock_to_frd.get(t)
            if fs:
                futures_symbol = fs
                break
        if futures_symbol is None:
            continue

        if futures_symbol not in futures_cache:
            fut = pd.read_parquet(
                futures_dir,
                filters=[('symbol', '==', futures_symbol), ('timestamp', '>=', pd.Timestamp(start_date, tz='America/New_York'))],
                columns=['timestamp', 'symbol', 'close'],
            ).sort_values('timestamp')
            fut['date'] = fut['timestamp'].dt.strftime('%Y-%m-%d')
            fut = fut.set_index('timestamp')
            futures_cache[futures_symbol] = fut[['date', 'close']]
        fut = futures_cache[futures_symbol]

        merged_fut = fut.merge(close_df.rename('domestic_close_time'), left_on='date', right_index=True)
        fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time', 'close']].apply(
            lambda x: x[x.index <= x['domestic_close_time'] + offset].iloc[-1]['close']
            if (x.index <= x['domestic_close_time'] + offset).any() else np.nan
        ).to_frame(name='fut_domestic_close')
        fut_series = fut['close'].astype(np.float32)

        baseline_full = {}
        baseline_by_date = {}
        for t in tickers:
            bp = baseline_signal_dir / f'ticker={t}' / 'data.parquet'
            if not bp.exists():
                continue
            bdf = pd.read_parquet(bp)
            baseline_full[t] = bdf
            baseline_by_date[t] = {d: g[['signal', 'date']] for d, g in bdf.groupby('date', sort=False)}

        ticker_parts = {t: [] for t in tickers}
        fallback_count = {t: 0 for t in tickers}

        for date_str in tqdm(sorted(close_df.index.tolist()), desc=f"{ex}"):
            date_ts = pd.Timestamp(date_str)
            if not (model_start <= date_ts <= model_end):
                for t in tickers:
                    fallback_count[t] += 1
                continue

            close_time = close_df.loc[date_str] + offset
            if date_str not in fut_domestic_close.index:
                for t in tickers:
                    fallback_count[t] += 1
                continue
            fut_close_price = float(fut_domestic_close.loc[date_str, 'fut_domestic_close'])
            if np.isnan(fut_close_price) or fut_close_price == 0:
                for t in tickers:
                    fallback_count[t] += 1
                continue

            beta_dates = sector_betas.index[sector_betas.index <= date_ts]
            if len(beta_dates) == 0:
                for t in tickers:
                    fallback_count[t] += 1
                continue
            beta_row = sector_betas.loc[beta_dates[-1]]

            for t in tickers:
                day_df = baseline_by_date.get(t, {}).get(date_str)
                if day_df is None or day_df.empty:
                    continue

                model_entry = get_model_for_date(models_by_ticker.get(t, []), date_ts)
                if model_entry is None:
                    fallback_count[t] += 1
                    continue

                hedge = adr_to_sector.get(t)
                if hedge is None:
                    fallback_count[t] += 1
                    continue

                if hedge not in sector_cache:
                    pp = sector_ohlcv_dir / f'ticker={hedge}' / 'data.parquet'
                    if not pp.exists():
                        sector_cache[hedge] = None
                    else:
                        sdf = pd.read_parquet(pp, columns=['Close', 'date'])
                        s = sdf['Close'].astype(np.float32)
                        if s.index.tz is None:
                            s.index = s.index.tz_localize('America/New_York')
                        sector_cache[hedge] = s

                s_series = sector_cache.get(hedge)
                if s_series is None:
                    fallback_count[t] += 1
                    continue

                day_series = s_series.loc[date_str:date_str]
                day_ffill = day_series.ffill()
                after = day_ffill[day_ffill.index >= close_time]
                if after.empty:
                    fallback_count[t] += 1
                    continue

                sector_close = float(after.iloc[0])
                if np.isnan(sector_close) or sector_close == 0:
                    fallback_count[t] += 1
                    continue

                beta_val = float(beta_row.get(hedge, 0.0))

                idx = day_df.index
                sector_prices = after.reindex(idx, method='ffill').to_numpy(dtype=np.float32)
                sector_ret = (sector_prices - sector_close) / sector_close
                fut_prices = fut_series.reindex(idx, method='ffill').to_numpy(dtype=np.float32)
                fut_ret = (fut_prices - fut_close_price) / fut_close_price
                x = np.nan_to_num(sector_ret - beta_val * fut_ret, nan=0.0, posinf=0.0, neginf=0.0)

                p = model_entry['w'] * x + model_entry['c']
                aug = day_df[['signal']].copy()
                aug['signal'] = aug['signal'].to_numpy(dtype=np.float32) + p
                aug['date'] = date_str
                ticker_parts[t].append(aug)
                augmented_any.add(t)

        for t in tickers:
            parts = ticker_parts[t]
            out_df = pd.concat(parts) if parts else pd.DataFrame(columns=['signal', 'date'])
            bdf = baseline_full.get(t)
            if bdf is not None:
                all_dates = set(bdf['date'].unique())
                out_dates = set(out_df['date'].unique()) if not out_df.empty else set()
                missing = all_dates - out_dates
                if missing:
                    out_df = pd.concat([out_df, bdf[bdf['date'].isin(missing)]]).sort_index()
            tdir = output_dir / f'ticker={t}'
            tdir.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(tdir / 'data.parquet')
            print(f"{t}: augmented={len(parts)}, fallback={fallback_count[t]}")

        gc.collect()

    # Copy all non-eligible baseline tickers
    all_adrs = adr_info['adr'].tolist()
    non_eligible = [t for t in all_adrs if t not in eligible]
    for t in non_eligible:
        src = baseline_signal_dir / f'ticker={t}' / 'data.parquet'
        if src.exists():
            d = output_dir / f'ticker={t}'
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, d / 'data.parquet')

    print(f"Done in {(time.perf_counter()-t0)/60:.2f} min; augmented={len(augmented_any)}, baseline_copied={len(non_eligible)}")


if __name__ == '__main__':
    main()
