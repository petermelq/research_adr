"""
Generate Ridge-augmented intraday signal for ADR arbitrage.

For Ridge-eligible ADRs:
  signal = baseline_signal + ridge_residual_pred

For non-Ridge ADRs:
  copy baseline signal from data/processed/futures_only_signal

This implementation is optimized for speed:
- Vectorized residual construction (matrix ops)
- Vectorized model inference using precomputed linear weights
- Baseline signals grouped by date once (no repeated per-date filtering)
- Russell data loaded once per batch as a wide matrix

Benchmark mode can be used for fast runtime estimation.
"""

import argparse
import gc
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from train_ridge_models import RidgeResidualModel  # noqa: F401

__script_dir__ = Path(__file__).parent.absolute()

RIDGE_ELIGIBLE = [
    'ARGX', 'ASML', 'AZN', 'BP', 'BTI', 'BUD', 'DEO', 'E', 'EQNR',
    'FMS', 'GMAB', 'GSK', 'HLN', 'IHG', 'NGG', 'NVS', 'RIO',
    'SHEL', 'SNY', 'TS', 'TTE', 'UL',
]

TIME_FUTURES_AFTER_CLOSE = {
    'XLON': pd.Timedelta('6min'),
    'XAMS': pd.Timedelta('6min'),
    'XPAR': pd.Timedelta('6min'),
    'XETR': pd.Timedelta('6min'),
    'XMIL': pd.Timedelta('6min'),
    'XBRU': pd.Timedelta('6min'),
    'XMAD': pd.Timedelta('6min'),
    'XHEL': pd.Timedelta('0min'),
    'XDUB': pd.Timedelta('0min'),
    'XOSL': pd.Timedelta('5min'),
    'XSTO': pd.Timedelta('0min'),
    'XSWX': pd.Timedelta('1min'),
    'XCSE': pd.Timedelta('0min'),
    'XTKS': pd.Timedelta('1min'),
    'XASX': pd.Timedelta('11min'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate ridge intraday signal')
    parser.add_argument('--benchmark-dates-per-exchange', type=int, default=0)
    parser.add_argument('--benchmark-exchanges', type=int, default=0)
    parser.add_argument('--date-batch-size', type=int, default=120)
    return parser.parse_args()


def _precompute_linear_params(model_obj):
    """
    Convert standardized linear model into equivalent raw-feature linear form.

    Original: y = ((x - mean) / scale) @ coef
    Equivalent: y = x @ (coef/scale) - (mean/scale) @ coef
    """
    coef = model_obj.model.coef_.astype(np.float32)
    scale = model_obj.scaler.scale_.astype(np.float32)
    mean = model_obj.scaler.mean_.astype(np.float32)

    safe_scale = np.where(scale == 0, 1.0, scale)
    w = coef / safe_scale
    c = -float(np.dot(mean / safe_scale, coef))
    return w, c


def load_ridge_models(model_dir):
    """
    Load ridge models and precompute linear params.

    Returns:
      models_by_ticker: dict[ticker] -> list of model dicts
      canonical_features: list[str]
      model_start: pd.Timestamp
      model_end: pd.Timestamp
    """
    models_by_ticker = {}
    canonical_features = None

    model_start = None
    model_end = None

    for ticker in RIDGE_ELIGIBLE:
        ticker_dir = model_dir / ticker
        if not ticker_dir.exists():
            continue

        entries = []
        for mf in sorted(ticker_dir.glob('*.pkl')):
            with open(mf, 'rb') as f:
                model_data = pickle.load(f)

            test_start, test_end = model_data['test_period']
            test_start = pd.Timestamp(test_start)
            test_end = pd.Timestamp(test_end)

            feature_names = model_data['feature_names']
            if canonical_features is None:
                canonical_features = feature_names

            w, c = _precompute_linear_params(model_data['model'])

            entries.append({
                'test_start': test_start,
                'test_end': test_end,
                'feature_names': feature_names,
                'w_raw': w,
                'c_raw': c,
            })

            if model_start is None or test_start < model_start:
                model_start = test_start
            if model_end is None or test_end > model_end:
                model_end = test_end

        if entries:
            models_by_ticker[ticker] = entries

    if canonical_features is None:
        canonical_features = []

    # If model feature orders differ, expand each model to canonical order once.
    canonical_pos = {fn: i for i, fn in enumerate(canonical_features)}
    n_features = len(canonical_features)
    for ticker, entries in models_by_ticker.items():
        for entry in entries:
            if entry['feature_names'] == canonical_features:
                entry['w'] = entry['w_raw']
            else:
                w_full = np.zeros(n_features, dtype=np.float32)
                for fn, wv in zip(entry['feature_names'], entry['w_raw']):
                    pos = canonical_pos.get(fn)
                    if pos is not None:
                        w_full[pos] = wv
                entry['w'] = w_full

    return models_by_ticker, canonical_features, model_start, model_end


def get_model_for_date(model_entries, date_ts):
    for entry in model_entries:
        if entry['test_start'] <= date_ts <= entry['test_end']:
            return entry
    return None


def load_russell_batch_matrix(russell_ohlcv_dir, tickers, dates_set):
    """
    Load batch Russell minute data as wide matrix.

    Returns DataFrame indexed by ET timestamps, columns=tickers, dtype=float32.
    """
    data = {}
    for ticker in tickers:
        parquet_path = russell_ohlcv_dir / f'ticker={ticker}' / 'data.parquet'
        if not parquet_path.exists():
            continue
        try:
            df = pd.read_parquet(parquet_path, columns=['Close', 'date'])
            df = df[df['date'].isin(dates_set)]
            if df.empty:
                continue
            s = df['Close'].astype(np.float32)
            if s.index.tz is None:
                s.index = s.index.tz_localize('America/New_York')
            data[ticker] = s
        except Exception:
            continue

    if not data:
        return pd.DataFrame()

    wide = pd.DataFrame(data).sort_index()
    return wide


def load_baseline_for_exchange(eligible_tickers, baseline_signal_dir):
    baseline_full = {}
    baseline_by_date = {}

    for ticker in eligible_tickers:
        bp = baseline_signal_dir / f'ticker={ticker}' / 'data.parquet'
        if not bp.exists():
            continue

        df = pd.read_parquet(bp)
        baseline_full[ticker] = df

        # Group once by date for O(1) per-date lookup later.
        date_map = {}
        for d, g in df.groupby('date', sort=False):
            date_map[d] = g[['signal', 'date']]
        baseline_by_date[ticker] = date_map

    return baseline_full, baseline_by_date


def process_exchange(
    exchange_mic,
    eligible_tickers,
    args,
    close_df,
    models_by_ticker,
    canonical_tickers,
    model_start,
    model_end,
    russell_betas_dir,
    stock_to_frd,
    futures_dir,
    baseline_signal_dir,
    russell_ohlcv_dir,
    output_dir,
    benchmark_mode,
    start_date,
    russell_batch_cache,
    futures_symbol_cache,
):
    offset = TIME_FUTURES_AFTER_CLOSE[exchange_mic]
    available_dates = sorted(close_df.index.tolist())

    betas_path = russell_betas_dir / f'{exchange_mic}.parquet'
    if not betas_path.exists():
        print('  Russell betas not found, skipping')
        return {'processed_dates': 0, 'covered_dates': 0, 'augmented_tickers': set()}

    russell_betas = pd.read_parquet(betas_path)
    russell_betas.index = pd.to_datetime(russell_betas.index)

    # futures symbol for this exchange: any eligible ticker with mapping
    futures_symbol = None
    for ticker in eligible_tickers:
        fs = stock_to_frd.get(ticker)
        if fs:
            futures_symbol = fs
            break
    if futures_symbol is None:
        print('  No futures mapping, skipping')
        return {'processed_dates': 0, 'covered_dates': 0, 'augmented_tickers': set()}

    if futures_symbol not in futures_symbol_cache:
        print(f'  Loading futures data for {futures_symbol}...')
        futures_df_raw = pd.read_parquet(
            futures_dir,
            filters=[
                ('symbol', '==', futures_symbol),
                ('timestamp', '>=', pd.Timestamp(start_date, tz='America/New_York')),
            ],
            columns=['timestamp', 'symbol', 'close'],
        )
        futures_df_raw = futures_df_raw.sort_values('timestamp')
        futures_df_raw['date'] = futures_df_raw['timestamp'].dt.strftime('%Y-%m-%d')
        futures_df_raw = futures_df_raw.set_index('timestamp')
        futures_symbol_cache[futures_symbol] = futures_df_raw[['date', 'close']]
    else:
        futures_df_raw = futures_symbol_cache[futures_symbol]

    merged_fut = futures_df_raw.merge(
        close_df.rename('domestic_close_time'), left_on='date', right_index=True
    )

    fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time', 'close']].apply(
        lambda x: x[x.index <= x['domestic_close_time'] + offset].iloc[-1]['close']
        if (x.index <= x['domestic_close_time'] + offset).any()
        else np.nan
    ).to_frame(name='fut_domestic_close')

    futures_close_series = futures_df_raw['close'].astype(np.float32)
    del futures_df_raw, merged_fut
    gc.collect()

    baseline_full, baseline_by_date = load_baseline_for_exchange(eligible_tickers, baseline_signal_dir)

    ticker_augmented = {t: [] for t in eligible_tickers}
    ticker_fallback_count = {t: 0 for t in eligible_tickers}

    covered_dates = [
        d for d in available_dates
        if model_start <= pd.Timestamp(d) <= model_end
    ]
    non_covered_dates = [
        d for d in available_dates
        if pd.Timestamp(d) < model_start or pd.Timestamp(d) > model_end
    ]

    if benchmark_mode and args.benchmark_dates_per_exchange > 0:
        covered_dates = covered_dates[:args.benchmark_dates_per_exchange]

    for t in eligible_tickers:
        ticker_fallback_count[t] += len(non_covered_dates)

    print(
        f'  {len(covered_dates)} dates with model coverage, '
        f'{len(non_covered_dates)} dates without (fallback)'
    )

    date_batch_size = max(1, args.date_batch_size)
    n_batches = (len(covered_dates) + date_batch_size - 1) // date_batch_size
    print(f'  Processing in {n_batches} batches of up to {date_batch_size} dates')

    ticker_to_col = {t: i for i, t in enumerate(canonical_tickers)}
    n_features = len(canonical_tickers)

    processed_dates = 0

    for batch_idx in range(n_batches):
        b0 = batch_idx * date_batch_size
        b1 = min(b0 + date_batch_size, len(covered_dates))
        batch_dates = covered_dates[b0:b1]
        batch_set = set(batch_dates)

        cache_key = tuple(batch_dates)
        if cache_key in russell_batch_cache:
            russell_df = russell_batch_cache[cache_key]
        else:
            russell_df = load_russell_batch_matrix(russell_ohlcv_dir, canonical_tickers, batch_set)
            russell_batch_cache[cache_key] = russell_df
            # Small LRU-like cap to avoid unbounded memory in long runs.
            if len(russell_batch_cache) > 4:
                first_key = next(iter(russell_batch_cache))
                del russell_batch_cache[first_key]

        if russell_df.empty:
            for t in eligible_tickers:
                ticker_fallback_count[t] += len(batch_dates)
            continue

        for date_str in tqdm(batch_dates, desc=f'  {exchange_mic} batch {batch_idx+1}/{n_batches}'):
            processed_dates += 1
            date_ts = pd.Timestamp(date_str)
            close_time = close_df.loc[date_str] + offset

            # Find active ticker/model/date tuples.
            active = []
            for ticker in eligible_tickers:
                day_df = baseline_by_date.get(ticker, {}).get(date_str)
                if day_df is None or day_df.empty:
                    continue

                model_entry = get_model_for_date(models_by_ticker.get(ticker, []), date_ts)
                if model_entry is None:
                    ticker_fallback_count[ticker] += 1
                    continue

                active.append((ticker, model_entry, day_df))

            if not active:
                continue

            day_matrix = russell_df.loc[date_str:date_str]
            # Forward-fill from the full trading day so the first post-close row
            # has the most recent known value for each Russell ticker.
            day_matrix_ffill = day_matrix.ffill()
            after_close = day_matrix_ffill[day_matrix_ffill.index >= close_time]
            if after_close.empty:
                for ticker, _, _ in active:
                    ticker_fallback_count[ticker] += 1
                continue
            russell_close = after_close.iloc[0].to_numpy(dtype=np.float32)
            valid_mask = ~np.isnan(russell_close)
            if not valid_mask.any():
                for ticker, _, _ in active:
                    ticker_fallback_count[ticker] += 1
                continue

            if date_str not in fut_domestic_close.index:
                for ticker, _, _ in active:
                    ticker_fallback_count[ticker] += 1
                continue

            fut_close_price = float(fut_domestic_close.loc[date_str, 'fut_domestic_close'])
            if np.isnan(fut_close_price) or fut_close_price == 0:
                for ticker, _, _ in active:
                    ticker_fallback_count[ticker] += 1
                continue

            beta_dates = russell_betas.index[russell_betas.index <= date_ts]
            if len(beta_dates) == 0:
                for ticker, _, _ in active:
                    ticker_fallback_count[ticker] += 1
                continue

            beta_row = russell_betas.loc[beta_dates[-1]].reindex(canonical_tickers).fillna(0.0)
            beta_vec = beta_row.to_numpy(dtype=np.float32)

            # Canonical prediction index: union of active baseline minute indices.
            pred_index = active[0][2].index
            for _, _, bd in active[1:]:
                if not bd.index.equals(pred_index):
                    pred_index = pred_index.union(bd.index)
            pred_index = pred_index.sort_values()

            # Build residual feature matrix on pred_index.
            valid_cols = np.where(valid_mask)[0]
            aligned_prices = after_close.iloc[:, valid_cols].reindex(pred_index, method='ffill')
            price_arr = aligned_prices.to_numpy(dtype=np.float32)
            close_arr = russell_close[valid_cols]

            returns_arr = (price_arr - close_arr) / close_arr

            fut_arr = futures_close_series.reindex(pred_index, method='ffill').to_numpy(dtype=np.float32)
            fut_ret = (fut_arr - fut_close_price) / fut_close_price

            residual_arr = returns_arr - fut_ret[:, None] * beta_vec[valid_cols][None, :]
            residual_arr = np.nan_to_num(residual_arr, nan=0.0, posinf=0.0, neginf=0.0)

            x_full = np.zeros((len(pred_index), n_features), dtype=np.float32)
            x_full[:, valid_cols] = residual_arr

            # Stack active model weights and infer in one matrix multiply.
            active_tickers = [a[0] for a in active]
            w_mat = np.column_stack([a[1]['w'] for a in active]).astype(np.float32)
            c_vec = np.array([a[1]['c_raw'] for a in active], dtype=np.float32)
            pred_mat = x_full @ w_mat + c_vec

            pred_df = pd.DataFrame(pred_mat, index=pred_index, columns=active_tickers)

            for ticker, _, baseline_date_df in active:
                p = pred_df[ticker].reindex(baseline_date_df.index, method='ffill').to_numpy(dtype=np.float32)

                augmented = baseline_date_df[['signal']].copy()
                augmented['signal'] = augmented['signal'].to_numpy(dtype=np.float32) + p
                augmented['date'] = date_str
                ticker_augmented[ticker].append(augmented)

        gc.collect()

    augmented_tickers = set()

    # Save exchange outputs.
    if not benchmark_mode:
        for ticker in eligible_tickers:
            parts = ticker_augmented[ticker]
            if parts:
                out_df = pd.concat(parts)
            else:
                out_df = pd.DataFrame(columns=['signal', 'date'])

            # Fill missing dates from baseline.
            baseline_df = baseline_full.get(ticker)
            if baseline_df is not None:
                all_dates = set(baseline_df['date'].unique())
                out_dates = set(out_df['date'].unique()) if not out_df.empty else set()
                missing = all_dates - out_dates
                if missing:
                    fallback = baseline_df[baseline_df['date'].isin(missing)]
                    out_df = pd.concat([out_df, fallback]).sort_index()

            tdir = output_dir / f'ticker={ticker}'
            tdir.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(tdir / 'data.parquet')
            augmented_tickers.add(ticker)

            print(f"  {ticker}: {len(parts)} dates augmented, {ticker_fallback_count[ticker]} fallback")

    del ticker_augmented, ticker_fallback_count, baseline_full, baseline_by_date
    del fut_domestic_close, futures_close_series, russell_betas
    gc.collect()

    return {
        'processed_dates': processed_dates,
        'covered_dates': len(covered_dates),
        'augmented_tickers': augmented_tickers,
    }


def main():
    args = parse_args()

    benchmark_mode = args.benchmark_dates_per_exchange > 0

    print('=' * 70)
    print('Ridge-Augmented Intraday Signal Generation')
    if benchmark_mode:
        print('BENCHMARK MODE ENABLED')
    print('=' * 70)

    params = load_params()
    start_date = params['start_date']
    end_date = params['end_date']
    print(f'Date range: {start_date} to {end_date}')

    data_dir = __script_dir__ / '..'
    futures_dir = data_dir / 'data' / 'processed' / 'futures' / 'converted_minute_bars'
    russell_ohlcv_dir = data_dir / 'data' / 'raw' / 'russell1000' / 'ohlcv-1m'
    russell_betas_dir = data_dir / 'data' / 'processed' / 'russell1000' / 'russell_betas'
    model_dir = data_dir / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'ridge'
    baseline_signal_dir = data_dir / 'data' / 'processed' / 'futures_only_signal'
    output_dir = data_dir / 'data' / 'processed' / 'index_russell_ridge_signal'

    # In non-benchmark mode, clear old partitions first.
    if not benchmark_mode:
        output_dir.mkdir(parents=True, exist_ok=True)
        for p in output_dir.glob('ticker=*'):
            shutil.rmtree(p, ignore_errors=True)

    adr_info = pd.read_csv(data_dir / 'data' / 'raw' / 'adr_info.csv')
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity', '')
    adr_tickers = adr_info['adr'].tolist()
    exchange_dict = adr_info.set_index('adr')['exchange'].to_dict()

    futures_symbols = pd.read_csv(data_dir / 'data' / 'raw' / 'futures_symbols.csv')
    adr_info['index_future_bbg'] = adr_info['index_future_bbg'].str.strip()
    futures_symbols['bloomberg_symbol'] = futures_symbols['bloomberg_symbol'].str.strip()
    merged_info = adr_info.merge(
        futures_symbols,
        left_on='index_future_bbg',
        right_on='bloomberg_symbol',
    )
    stock_to_frd = merged_info.set_index(
        'adr_x' if 'adr_x' in merged_info.columns else 'adr'
    )['first_rate_symbol'].to_dict()

    print('\nLoading Ridge models...')
    models_by_ticker, canonical_features, model_start, model_end = load_ridge_models(model_dir)
    print(f'Loaded models for {len(models_by_ticker)} tickers')
    print(f'Total Russell tickers needed: {len(canonical_features)}')
    print(f'Model date range: {model_start} to {model_end}')

    canonical_tickers = [fn.replace('russell_', '', 1) for fn in canonical_features]

    eligible_by_exchange = {}
    for ticker in RIDGE_ELIGIBLE:
        ex = exchange_dict.get(ticker)
        if ex:
            eligible_by_exchange.setdefault(ex, []).append(ticker)

    exchanges = sorted(eligible_by_exchange.keys())
    if benchmark_mode and args.benchmark_exchanges > 0:
        exchanges = exchanges[:args.benchmark_exchanges]

    print('\nRidge-eligible tickers by exchange:')
    for ex in exchanges:
        print(f'  {ex}: {eligible_by_exchange[ex]}')

    exchange_close_times = {}
    for ex in exchanges:
        close_df = (
            mcal.get_calendar(ex)
            .schedule(start_date=start_date, end_date=end_date)['market_close']
            .dt.tz_convert('America/New_York')
        ).rename('domestic_close_time')
        close_df.index = close_df.index.astype(str)
        exchange_close_times[ex] = close_df

    overall_augmented = set()
    total_processed_dates = 0
    total_covered_dates = 0
    russell_batch_cache = {}
    futures_symbol_cache = {}

    t0 = time.perf_counter()

    for ex in exchanges:
        print(f"\n{'=' * 60}")
        print(f'Processing exchange: {ex}')
        print(f"{'=' * 60}")

        stats = process_exchange(
            exchange_mic=ex,
            eligible_tickers=eligible_by_exchange[ex],
            args=args,
            close_df=exchange_close_times[ex],
            models_by_ticker=models_by_ticker,
            canonical_tickers=canonical_tickers,
            model_start=model_start,
            model_end=model_end,
            russell_betas_dir=russell_betas_dir,
            stock_to_frd=stock_to_frd,
            futures_dir=futures_dir,
            baseline_signal_dir=baseline_signal_dir,
            russell_ohlcv_dir=russell_ohlcv_dir,
            output_dir=output_dir,
            benchmark_mode=benchmark_mode,
            start_date=start_date,
            russell_batch_cache=russell_batch_cache,
            futures_symbol_cache=futures_symbol_cache,
        )

        overall_augmented.update(stats['augmented_tickers'])
        total_processed_dates += stats['processed_dates']
        total_covered_dates += stats['covered_dates']

    elapsed = time.perf_counter() - t0

    if benchmark_mode:
        print(f"\n{'=' * 70}")
        print('Benchmark complete!')
        print(f'  Elapsed seconds: {elapsed:.2f}')
        print(f'  Processed covered dates: {total_processed_dates}')
        print(f'  Sample covered dates: {total_covered_dates}')
        if total_processed_dates > 0 and total_covered_dates > 0:
            # Estimate full run over all covered dates in selected exchanges.
            est_same_scope_sec = elapsed * (total_covered_dates / total_processed_dates)
            print(f'  Estimated full time for selected exchanges: {est_same_scope_sec / 60:.2f} min')
        print(f"{'=' * 70}")
        return

    # Copy non-ridge tickers from baseline.
    non_ridge_tickers = [t for t in adr_tickers if t not in RIDGE_ELIGIBLE]
    print(f'\nCopying baseline signal for {len(non_ridge_tickers)} non-Ridge tickers...')
    for ticker in non_ridge_tickers:
        src_path = baseline_signal_dir / f'ticker={ticker}' / 'data.parquet'
        dst_dir = output_dir / f'ticker={ticker}'
        if src_path.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_dir / 'data.parquet')

    # Copy fallback for eligible tickers if any skipped.
    for ticker in RIDGE_ELIGIBLE:
        if ticker in overall_augmented:
            continue
        src_path = baseline_signal_dir / f'ticker={ticker}' / 'data.parquet'
        dst_dir = output_dir / f'ticker={ticker}'
        if src_path.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_dir / 'data.parquet')
            print(f'  Copied baseline for {ticker} (no augmentation)')

    print(f"\n{'=' * 70}")
    print('Signal generation complete!')
    print(f'  Output directory: {output_dir}')
    print(f'  Ridge-augmented: {len(overall_augmented)} tickers')
    print(f'  Baseline copied: {len(non_ridge_tickers)} tickers')
    print(f'  Elapsed: {elapsed / 60:.2f} min')
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
