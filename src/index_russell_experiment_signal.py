"""Generate intraday signal for experiment models trained on Russell residual features."""

import argparse
import gc
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from tqdm import tqdm

from utils import load_params

TIME_FUTURES_AFTER_CLOSE = {
    'XLON': pd.Timedelta('6min'), 'XAMS': pd.Timedelta('6min'), 'XPAR': pd.Timedelta('6min'),
    'XETR': pd.Timedelta('6min'), 'XMIL': pd.Timedelta('6min'), 'XBRU': pd.Timedelta('6min'),
    'XMAD': pd.Timedelta('6min'), 'XHEL': pd.Timedelta('0min'), 'XDUB': pd.Timedelta('0min'),
    'XOSL': pd.Timedelta('5min'), 'XSTO': pd.Timedelta('0min'), 'XSWX': pd.Timedelta('1min'),
    'XCSE': pd.Timedelta('0min'), 'XTKS': pd.Timedelta('1min'), 'XASX': pd.Timedelta('11min'),
    'XHKG': pd.Timedelta('10min'),
}
ASIA_EXCHANGES = {"XTKS", "XASX", "XHKG", "XSES", "XSHG", "XSHE"}


class _LegacyRidgeResidualModel:
    """Placeholder class for unpickling legacy ridge artifacts."""
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-kind', choices=['ridge', 'pcr', 'robust_pcr', 'elasticnet', 'lasso', 'pls', 'rf', 'rrr', 'huber'], required=True)
    p.add_argument(
        '--eval-times-only',
        action='store_true',
        help='Only compute/persist signals at 30-minute evaluation times (13:00-15:30 ET).',
    )
    p.add_argument('--exchange-workers', type=int, default=1, help='Parallel workers across exchanges.')
    p.add_argument('--only-exchange', default=None, help='Run only a single exchange (internal parallel mode).')
    p.add_argument('--skip-clean', action='store_true', help='Do not clean output dir before writing.')
    p.add_argument('--log-batch-timing', action='store_true', help='Log per-batch load/compute timing checkpoints.')
    p.add_argument('--max-covered-dates', type=int, default=None, help='Optional cap on covered dates for quick profiling.')
    p.add_argument('--min-inference-date', type=str, default=None, help='Only generate signal on/after this date (YYYY-MM-DD).')
    return p.parse_args()


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__' and name == 'RidgeResidualModel':
            return _LegacyRidgeResidualModel
        return super().find_class(module, name)


def _safe_load_pickle(path):
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except AttributeError as e:
            msg = str(e)
            if 'RidgeResidualModel' not in msg:
                raise
    with open(path, 'rb') as f:
        return _CompatUnpickler(f).load()


def _augment_legacy_linear_model(md):
    if not isinstance(md, dict) or md.get('w_raw') is not None:
        return md
    legacy = md.get('model')
    if legacy is None or not hasattr(legacy, 'scaler') or not hasattr(legacy, 'model'):
        return md
    scaler = legacy.scaler
    reg = legacy.model
    if not hasattr(reg, 'coef_'):
        return md

    coef = np.asarray(reg.coef_, dtype=np.float32)
    scale = np.asarray(getattr(scaler, 'scale_', np.ones_like(coef)), dtype=np.float32)
    mean = np.asarray(getattr(scaler, 'mean_', np.zeros_like(coef)), dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    w = coef / scale
    c = -float(np.dot(mean / scale, coef))

    md['kind'] = 'linear'
    md['w_raw'] = w
    md['c_raw'] = c
    if 'feature_names' not in md or md['feature_names'] is None:
        md['feature_names'] = getattr(legacy, 'feature_names', [])
    return md


def load_models(model_dir):
    models = {}
    canonical_features = None
    canonical_feature_set = set()
    model_start = None
    model_end = None

    for tdir in sorted(model_dir.iterdir()):
        if not tdir.is_dir():
            continue
        ticker = tdir.name
        entries = []
        for mf in sorted(tdir.glob('*.pkl')):
            md = _augment_legacy_linear_model(_safe_load_pickle(mf))
            ts, te = map(pd.Timestamp, md['test_period'])
            feature_names = md['feature_names']
            if feature_names is None:
                feature_names = []
            canonical_feature_set.update(feature_names)

            entries.append({
                'test_start': ts,
                'test_end': te,
                'kind': md.get('kind', 'linear'),
                'feature_names': feature_names,
                'w_raw': md.get('w_raw'),
                'c_raw': md.get('c_raw', 0.0),
                'model': md.get('model'),
            })

            model_start = ts if model_start is None or ts < model_start else model_start
            model_end = te if model_end is None or te > model_end else model_end

        if entries:
            models[ticker] = entries

    if canonical_features is None:
        # Use the union of all model feature names as canonical space so no feature
        # used by any artifact is dropped during inference remapping.
        canonical_features = sorted(canonical_feature_set)
    missing_feature_refs = 0

    canonical_pos = {fn: i for i, fn in enumerate(canonical_features)}
    n_features = len(canonical_features)
    global_tickers = [fn.replace('russell_', '', 1) for fn in canonical_features]
    global_ticker_pos = {t: i for i, t in enumerate(global_tickers)}

    for ticker, entries in models.items():
        for e in entries:
            feat_tickers = [fn.replace('russell_', '', 1) for fn in e['feature_names']]
            e['feature_tickers'] = feat_tickers
            e['feature_pos_global'] = np.array(
                [global_ticker_pos[t] for t in feat_tickers if t in global_ticker_pos], dtype=np.int32
            )
            if len(e['feature_pos_global']) != len(feat_tickers):
                missing_feature_refs += len(feat_tickers) - len(e['feature_pos_global'])
            if e['kind'] == 'linear':
                w_raw = np.asarray(e['w_raw'], dtype=np.float32)
                if e['feature_names'] == canonical_features:
                    e['w'] = w_raw
                else:
                    w = np.zeros(n_features, dtype=np.float32)
                    for fn, val in zip(e['feature_names'], w_raw):
                        pos = canonical_pos.get(fn)
                        if pos is not None:
                            w[pos] = val
                    e['w'] = w
                e['c'] = float(e['c_raw'])

    if missing_feature_refs > 0:
        print(f'WARNING: missing canonical mapping refs: {missing_feature_refs}', flush=True)
    return models, canonical_features, model_start, model_end, global_ticker_pos


def get_model_for_date(entries, d):
    for e in entries:
        if e['test_start'] <= d <= e['test_end']:
            return e
    return None


def load_russell_batch_matrix(russell_ohlcv_dir, tickers, dates_set, needed_index_by_date=None):
    data = {}
    if dates_set:
        date_min = min(dates_set)
        date_max = max(dates_set)
        range_start = pd.Timestamp(f'{date_min} 00:00:00')
        range_end = pd.Timestamp(f'{date_max} 23:59:59')
    else:
        range_start = None
        range_end = None
    # Fast path: read all requested tickers for the batch in one parquet scan.
    # This avoids hundreds of per-ticker file opens that dominate runtime.
    if range_start is not None and tickers:
        try:
            bulk = pd.read_parquet(
                russell_ohlcv_dir,
                columns=['Close', 'DateTime', 'ticker'],
                filters=[('ticker', 'in', list(tickers)), ('DateTime', '>=', range_start), ('DateTime', '<=', range_end)],
            )
            if not bulk.empty:
                bulk_idx = pd.to_datetime(bulk['DateTime'], errors='coerce')
                good = ~bulk_idx.isna()
                bulk = bulk.loc[good].copy()
                bulk_idx = bulk_idx.loc[good]
                if bulk_idx.dt.tz is None:
                    bulk_idx = bulk_idx.dt.tz_localize('America/New_York')
                else:
                    bulk_idx = bulk_idx.dt.tz_convert('America/New_York')
                bulk['DateTime'] = bulk_idx.values
                bulk['ticker'] = bulk['ticker'].astype(str)
                wide = (
                    bulk.set_index(['DateTime', 'ticker'])['Close']
                    .unstack('ticker')
                    .sort_index()
                    .astype(np.float32)
                )
                # Keep stable feature order.
                keep_cols = [t for t in tickers if t in wide.columns]
                wide = wide[keep_cols]
                if needed_index_by_date:
                    out_parts = []
                    day_key = wide.index.strftime('%Y-%m-%d')
                    for d, want_idx in needed_index_by_date.items():
                        if len(want_idx) == 0:
                            continue
                        day_df = wide[day_key == d]
                        if day_df.empty:
                            continue
                        out_parts.append(day_df.reindex(want_idx, method='ffill'))
                    if out_parts:
                        return pd.concat(out_parts).sort_index()
                else:
                    return wide
        except Exception:
            # Fall back to conservative per-ticker load path below.
            pass

    for ticker in tickers:
        parquet_path = russell_ohlcv_dir / f'ticker={ticker}' / 'data.parquet'
        if not parquet_path.exists():
            continue
        try:
            # Russell minute bars are expected to have DateTime + Close semantics.
            # Use DatetimeIndex as source of truth and derive date keys from it.
            if range_start is not None:
                try:
                    df = pd.read_parquet(
                        parquet_path,
                        columns=['Close', 'DateTime'],
                        filters=[('DateTime', '>=', range_start), ('DateTime', '<=', range_end)],
                    )
                except Exception:
                    df = pd.read_parquet(parquet_path, columns=['Close'])
            else:
                df = pd.read_parquet(parquet_path, columns=['Close'])
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index
            elif 'DateTime' in df.columns:
                idx = pd.to_datetime(df['DateTime'], errors='coerce')
            else:
                continue
            if idx.tz is None:
                idx = idx.tz_localize('America/New_York')
            else:
                idx = idx.tz_convert('America/New_York')
            df = df.copy()
            df.index = idx
            if range_start is None or needed_index_by_date is not None:
                # Exact date filtering is needed when no parquet range filter was applied
                # or when we are in eval-times mode and need precise day subsets.
                day_key = df.index.strftime('%Y-%m-%d')
                df = df[day_key.isin(dates_set)]
                if df.empty:
                    continue
            if needed_index_by_date:
                # Keep only required timestamps per date via forward-filled reindex.
                out_parts = []
                for d, want_idx in needed_index_by_date.items():
                    if len(want_idx) == 0:
                        continue
                    day_df = df[df.index.strftime('%Y-%m-%d') == d]
                    if day_df.empty:
                        continue
                    g = pd.Series(day_df['Close'].to_numpy(dtype=np.float32), index=day_df.index)
                    s = g.reindex(want_idx, method='ffill')
                    out_parts.append(s)
                if not out_parts:
                    continue
                s = pd.concat(out_parts).sort_index()
            else:
                s = df['Close'].astype(np.float32)
            data[ticker] = s
        except Exception:
            continue

    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).sort_index()


def _asof_price_by_date(fut_index_ns, fut_dates_ns, fut_close, target_time_ns, target_date_ns, side):
    """Vectorized asof lookup constrained to same normalized date."""
    pos = np.searchsorted(fut_index_ns, target_time_ns, side=side)
    if side == 'right':
        pos = pos - 1
    valid = (pos >= 0) & (pos < len(fut_index_ns))
    out = np.full(len(target_time_ns), np.nan, dtype=np.float64)
    if valid.any():
        vv = np.where(valid)[0]
        p = pos[vv]
        same_day = fut_dates_ns[p] == target_date_ns[vv]
        if same_day.any():
            take = vv[same_day]
            out[take] = fut_close[pos[take]]
    return out


def _prepare_exchange_state(
    ex,
    tickers,
    close_df,
    stock_to_frd,
    model_start,
    model_end,
    max_covered_dates,
    min_inference_date,
    russell_betas_dir,
    futures_dir,
    futures_raw_cache,
):
    """Prepare per-exchange static inputs once so batch loop only does feature loading/prediction."""
    offset = TIME_FUTURES_AFTER_CLOSE[ex]
    betas_path = russell_betas_dir / f'{ex}.parquet'
    if not betas_path.exists():
        return None
    russell_betas = pd.read_parquet(betas_path)
    russell_betas.index = pd.to_datetime(russell_betas.index)

    futures_symbol = None
    for t in tickers:
        fs = stock_to_frd.get(t)
        if fs:
            futures_symbol = fs
            break
    if futures_symbol is None:
        return None

    # Cache futures minute series by symbol; many exchanges share the same hedge future.
    if futures_symbol not in futures_raw_cache:
        fut = pd.read_parquet(
            futures_dir,
            filters=[('symbol', '==', futures_symbol), ('timestamp', '>=', pd.Timestamp(load_params()['start_date'], tz='America/New_York'))],
            columns=['timestamp', 'symbol', 'close'],
        ).sort_values('timestamp')
        fut = fut.set_index('timestamp')
        # Avoid expensive per-row strftime over multi-million-row futures history.
        # Keep a normalized NY-timestamp date key for joins/groupbys.
        fut['date'] = fut.index.tz_convert('America/New_York').normalize()
        futures_raw_cache[futures_symbol] = fut
    futures_df_raw = futures_raw_cache[futures_symbol]

    close_dates = pd.to_datetime(close_df.index).tz_localize('America/New_York').normalize()
    close_times = pd.to_datetime(close_df.to_numpy())
    us_open_times = pd.to_datetime(close_df.index).tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=35)

    # Vectorized same-day asof selection for futures base prices.
    fut_index = futures_df_raw.index
    fut_index_ns = fut_index.values.astype('datetime64[ns]').astype(np.int64)
    fut_dates_ns = futures_df_raw['date'].values.astype('datetime64[ns]').astype(np.int64)
    fut_close = futures_df_raw['close'].to_numpy(dtype=np.float64)
    target_date_ns = close_dates.values.astype('datetime64[ns]').astype(np.int64)

    domestic_target_ns = (close_times + offset).values.astype('datetime64[ns]').astype(np.int64)
    fut_dom_vals = _asof_price_by_date(
        fut_index_ns=fut_index_ns,
        fut_dates_ns=fut_dates_ns,
        fut_close=fut_close,
        target_time_ns=domestic_target_ns,
        target_date_ns=target_date_ns,
        side='right',
    )
    fut_domestic_close = pd.DataFrame({'fut_domestic_close': fut_dom_vals}, index=close_dates)

    us_open_target_ns = us_open_times.values.astype('datetime64[ns]').astype(np.int64)
    fut_open_vals = _asof_price_by_date(
        fut_index_ns=fut_index_ns,
        fut_dates_ns=fut_dates_ns,
        fut_close=fut_close,
        target_time_ns=us_open_target_ns,
        target_date_ns=target_date_ns,
        side='left',
    )
    fut_us_open = pd.DataFrame({'fut_us_open': fut_open_vals}, index=close_dates)
    # Downstream date keys are YYYY-MM-DD strings.
    fut_domestic_close.index = fut_domestic_close.index.strftime('%Y-%m-%d')
    fut_us_open.index = fut_us_open.index.strftime('%Y-%m-%d')
    futures_close_series = futures_df_raw['close'].astype(np.float32)

    available_dates = sorted(close_df.index.tolist())
    covered_dates = [d for d in available_dates if model_start <= pd.Timestamp(d) <= model_end]
    if min_inference_date is not None:
        min_ts = pd.Timestamp(min_inference_date)
        covered_dates = [d for d in covered_dates if pd.Timestamp(d) >= min_ts]
    if max_covered_dates is not None:
        covered_dates = covered_dates[: max(0, int(max_covered_dates))]

    return {
        'exchange': ex,
        'tickers': tickers,
        'offset': offset,
        'close_df': close_df,
        'russell_betas': russell_betas,
        'fut_domestic_close': fut_domestic_close,
        'fut_us_open': fut_us_open,
        'futures_close_series': futures_close_series,
        'covered_dates': covered_dates,
        'covered_set': set(covered_dates),
        'ticker_pred': {t: [] for t in tickers},
    }


def _run_exchange(
    ex,
    tickers,
    args,
    close_df,
    stock_to_frd,
    models_by_ticker,
    canonical_tickers,
    global_ticker_pos,
    model_start,
    model_end,
    russell_betas_dir,
    futures_dir,
    russell_ohlcv_dir,
    eval_times,
    show_progress=True,
):
    offset = TIME_FUTURES_AFTER_CLOSE[ex]

    betas_path = russell_betas_dir / f'{ex}.parquet'
    if not betas_path.exists():
        return {}
    russell_betas = pd.read_parquet(betas_path)
    russell_betas.index = pd.to_datetime(russell_betas.index)

    futures_symbol = None
    for t in tickers:
        fs = stock_to_frd.get(t)
        if fs:
            futures_symbol = fs
            break
    if futures_symbol is None:
        return {}

    futures_df_raw = pd.read_parquet(
        futures_dir,
        filters=[('symbol', '==', futures_symbol), ('timestamp', '>=', pd.Timestamp(load_params()['start_date'], tz='America/New_York'))],
        columns=['timestamp', 'symbol', 'close'],
    ).sort_values('timestamp')
    futures_df_raw['date'] = futures_df_raw['timestamp'].dt.strftime('%Y-%m-%d')
    futures_df_raw = futures_df_raw.set_index('timestamp')

    merged_fut = futures_df_raw.merge(close_df.rename('domestic_close_time'), left_on='date', right_index=True)
    fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time', 'close']].apply(
        lambda x: x[x.index <= x['domestic_close_time'] + offset].iloc[-1]['close']
        if (x.index <= x['domestic_close_time'] + offset).any() else np.nan
    ).to_frame(name='fut_domestic_close')
    us_open_by_date = pd.DataFrame(
        {
            'us_open_time': (
                pd.to_datetime(close_df.index)
                .tz_localize('America/New_York')
                + pd.Timedelta(hours=9, minutes=35)
            )
        },
        index=close_df.index,
    )
    merged_fut_open = futures_df_raw.merge(us_open_by_date, left_on='date', right_index=True)
    fut_us_open = merged_fut_open.groupby('date')[['us_open_time', 'close']].apply(
        lambda x: x[x.index >= x['us_open_time']].iloc[0]['close']
        if (x.index >= x['us_open_time']).any() else np.nan
    ).to_frame(name='fut_us_open')
    futures_close_series = futures_df_raw['close'].astype(np.float32)

    ticker_pred = {t: [] for t in tickers}
    available_dates = sorted(close_df.index.tolist())
    covered_dates = [d for d in available_dates if model_start <= pd.Timestamp(d) <= model_end]
    if args.min_inference_date is not None:
        min_ts = pd.Timestamp(args.min_inference_date)
        covered_dates = [d for d in covered_dates if pd.Timestamp(d) >= min_ts]
    if args.max_covered_dates is not None:
        covered_dates = covered_dates[: max(0, int(args.max_covered_dates))]

    # Keep full-mode batches moderate to avoid OOM from wide in-memory Russell matrices.
    batch_size = 180 if not args.eval_times_only else 120
    n_batches = (len(covered_dates) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch_t0 = time.perf_counter()
        b0 = b * batch_size
        b1 = min(b0 + batch_size, len(covered_dates))
        batch_dates = covered_dates[b0:b1]
        batch_set = set(batch_dates)
        batch_start = pd.Timestamp(batch_dates[0])
        batch_end = pd.Timestamp(batch_dates[-1])
        needed_index_by_date = None
        if args.eval_times_only:
            needed_index_by_date = {}
            for d in batch_dates:
                base_ts = (
                    pd.Timestamp(f"{d} 09:35:00", tz='America/New_York')
                    if ex in ASIA_EXCHANGES
                    else close_df.loc[d] + offset
                )
                eval_idx = pd.DatetimeIndex(
                    [pd.Timestamp(f"{d} {t}:00", tz='America/New_York') for t in sorted(eval_times)]
                )
                needed_index_by_date[d] = eval_idx.union(pd.DatetimeIndex([base_ts])).sort_values()
        batch_feature_tickers = set()
        for ticker in tickers:
            for m in models_by_ticker.get(ticker, []):
                if m['test_end'] < batch_start or m['test_start'] > batch_end:
                    continue
                batch_feature_tickers.update(m['feature_tickers'])
        batch_tickers = [t for t in canonical_tickers if t in batch_feature_tickers]
        if not batch_tickers:
            if args.log_batch_timing:
                print(f'[{ex}] batch {b+1}/{n_batches}: no batch tickers (dates={len(batch_dates)})', flush=True)
            continue
        batch_global_pos = np.array([global_ticker_pos[t] for t in batch_tickers], dtype=np.int32)

        load_t0 = time.perf_counter()
        russell_df = load_russell_batch_matrix(
            russell_ohlcv_dir, batch_tickers, batch_set, needed_index_by_date=needed_index_by_date
        )
        load_sec = time.perf_counter() - load_t0
        if russell_df.empty:
            if args.log_batch_timing:
                print(
                    f'[{ex}] batch {b+1}/{n_batches}: empty russell_df '
                    f'(dates={len(batch_dates)} tickers={len(batch_tickers)} load={load_sec:.2f}s)',
                    flush=True,
                )
            continue

        compute_t0 = time.perf_counter()
        batch_active_dates = 0
        batch_signal_rows = 0
        iterator = tqdm(batch_dates, desc=f'{ex} {args.model_kind} {b+1}/{n_batches}', disable=not show_progress)
        for date_str in iterator:
            date_ts = pd.Timestamp(date_str)
            close_time = close_df.loc[date_str] + offset
            us_open_time = pd.Timestamp(f"{date_str} 09:35:00", tz="America/New_York")

            active = []
            for ticker in tickers:
                model_entry = get_model_for_date(models_by_ticker.get(ticker, []), date_ts)
                if model_entry is not None:
                    active.append((ticker, model_entry))

            if not active:
                continue
            batch_active_dates += 1

            day_matrix = russell_df.loc[date_str:date_str]
            day_matrix_ffill = day_matrix.ffill()

            if ex in ASIA_EXCHANGES:
                base_slice = day_matrix_ffill[day_matrix_ffill.index >= us_open_time]
            else:
                base_slice = day_matrix_ffill[day_matrix_ffill.index >= close_time]

            if base_slice.empty:
                continue

            russell_base = base_slice.iloc[0].to_numpy(dtype=np.float32)
            valid_mask = ~np.isnan(russell_base)
            if not valid_mask.any():
                continue

            if ex in ASIA_EXCHANGES:
                if date_str not in fut_us_open.index:
                    continue
                fut_base_price = float(fut_us_open.loc[date_str, 'fut_us_open'])
            else:
                if date_str not in fut_domestic_close.index:
                    continue
                fut_base_price = float(fut_domestic_close.loc[date_str, 'fut_domestic_close'])

            if np.isnan(fut_base_price) or fut_base_price == 0:
                continue

            beta_dates = russell_betas.index[russell_betas.index <= date_ts]
            if len(beta_dates) == 0:
                continue
            beta_row = russell_betas.loc[beta_dates[-1]].reindex(batch_tickers).fillna(0.0)
            beta_vec = beta_row.to_numpy(dtype=np.float32)

            if args.eval_times_only:
                pred_index = pd.DatetimeIndex(
                    [pd.Timestamp(f"{date_str} {t}:00", tz='America/New_York') for t in sorted(eval_times)]
                ).sort_values()
            else:
                pred_index = base_slice.index.sort_values()
            if ex in ASIA_EXCHANGES:
                pred_index = pred_index[pred_index >= us_open_time]
            if len(pred_index) == 0:
                continue

            valid_cols = np.where(valid_mask)[0]
            valid_global_cols = batch_global_pos[valid_cols]
            aligned_prices = base_slice.iloc[:, valid_cols].reindex(pred_index, method='ffill')
            price_arr = aligned_prices.to_numpy(dtype=np.float32)
            base_arr = russell_base[valid_cols]
            returns_arr = (price_arr - base_arr) / base_arr

            fut_arr = futures_close_series.reindex(pred_index, method='ffill').to_numpy(dtype=np.float32)
            fut_ret = (fut_arr - fut_base_price) / fut_base_price

            residual_arr = returns_arr - fut_ret[:, None] * beta_vec[valid_cols][None, :]
            residual_arr = np.nan_to_num(residual_arr, nan=0.0, posinf=0.0, neginf=0.0)
            x_full = np.zeros((len(pred_index), len(canonical_tickers)), dtype=np.float32)
            x_full[:, valid_global_cols] = residual_arr

            linear_active = [(t, m) for t, m in active if m['kind'] == 'linear']
            rf_active = [(t, m) for t, m in active if m['kind'] != 'linear']

            if linear_active:
                ltickers = [a[0] for a in linear_active]
                w_mat = np.column_stack([a[1]['w'][valid_global_cols] for a in linear_active]).astype(np.float32)
                c_vec = np.array([a[1]['c'] for a in linear_active], dtype=np.float32)
                pred_mat = residual_arr @ w_mat + c_vec
                pred_df = pd.DataFrame(pred_mat, index=pred_index, columns=ltickers)
                for ticker, _ in linear_active:
                    out_df = pd.DataFrame({'signal': pred_df[ticker].to_numpy(dtype=np.float32)}, index=pred_index)
                    out_df['date'] = date_str
                    ticker_pred[ticker].append(out_df)
                    batch_signal_rows += len(out_df)

            for ticker, mentry in rf_active:
                X_pred = x_full[:, mentry['feature_pos_global']]
                p = mentry['model'].predict(X_pred)
                out_df = pd.DataFrame({'signal': np.asarray(p, dtype=np.float32)}, index=pred_index)
                out_df['date'] = date_str
                ticker_pred[ticker].append(out_df)
                batch_signal_rows += len(out_df)

        compute_sec = time.perf_counter() - compute_t0
        if args.log_batch_timing:
            print(
                f'[{ex}] batch {b+1}/{n_batches}: dates={len(batch_dates)} tickers={len(batch_tickers)} '
                f'rows={len(russell_df)} cols={len(russell_df.columns)} '
                f'active_dates={batch_active_dates} signal_rows={batch_signal_rows} '
                f'load={load_sec:.2f}s compute={compute_sec:.2f}s total={(time.perf_counter()-batch_t0):.2f}s',
                flush=True,
            )
        gc.collect()

    out = {}
    for ticker in tickers:
        parts = ticker_pred[ticker]
        out[ticker] = pd.concat(parts) if parts else pd.DataFrame(columns=['signal', 'date'])
    del futures_df_raw, merged_fut, merged_fut_open, fut_domestic_close, fut_us_open, futures_close_series, russell_betas
    gc.collect()
    return out


def main():
    args = parse_args()
    eval_times = {'13:00', '13:30', '14:00', '14:30', '15:00', '15:30'}
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    params = load_params()
    start_date = params['start_date']
    end_date = params['end_date']

    data_dir = Path('.')
    futures_dir = data_dir / 'data' / 'processed' / 'futures' / 'converted_minute_bars'
    russell_ohlcv_dir = data_dir / 'data' / 'raw' / 'russell1000' / 'ohlcv-1m'
    russell_betas_dir = data_dir / 'data' / 'processed' / 'russell1000' / 'russell_betas'

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_clean:
        for p in output_dir.glob('ticker=*'):
            shutil.rmtree(p, ignore_errors=True)

    adr_info = pd.read_csv(data_dir / 'data' / 'raw' / 'adr_info.csv')
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    exchange_dict = adr_info.set_index('adr')['exchange'].to_dict()

    futures_symbols = pd.read_csv(data_dir / 'data' / 'raw' / 'futures_symbols.csv')
    adr_info['index_future_bbg'] = adr_info['index_future_bbg'].str.strip()
    futures_symbols['bloomberg_symbol'] = futures_symbols['bloomberg_symbol'].str.strip()
    merged_info = adr_info.merge(futures_symbols, left_on='index_future_bbg', right_on='bloomberg_symbol')
    stock_to_frd = merged_info.set_index('adr')['first_rate_symbol'].to_dict()

    models_by_ticker, canonical_features, model_start, model_end, global_ticker_pos = load_models(model_dir)
    canonical_tickers = [fn.replace('russell_', '', 1) for fn in canonical_features]

    eligible_by_exchange = {}
    modeled_tickers = sorted(models_by_ticker.keys())
    for t in modeled_tickers:
        ex = exchange_dict.get(t)
        if ex:
            eligible_by_exchange.setdefault(ex, []).append(t)
    if args.only_exchange is not None:
        eligible_by_exchange = {args.only_exchange: eligible_by_exchange.get(args.only_exchange, [])}

    exchange_close_times = {}
    for ex in sorted(eligible_by_exchange.keys()):
        close_df = (
            mcal.get_calendar(ex).schedule(start_date=start_date, end_date=end_date)['market_close']
            .dt.tz_convert('America/New_York')
        )
        close_df.index = close_df.index.astype(str)
        exchange_close_times[ex] = close_df

    overall_augmented = set()
    t0 = time.perf_counter()

    exchange_items = sorted(eligible_by_exchange.items())
    # Shared-batch path: load Russell minute data once per global batch and reuse across exchanges.
    if args.only_exchange is None:
        futures_raw_cache = {}
        exchange_states = {}
        for ex, tickers in exchange_items:
            st = _prepare_exchange_state(
                ex,
                tickers,
                exchange_close_times[ex],
                stock_to_frd,
                model_start,
                model_end,
                args.max_covered_dates,
                args.min_inference_date,
                russell_betas_dir,
                futures_dir,
                futures_raw_cache,
            )
            if st is not None and st['covered_dates']:
                exchange_states[ex] = st

        if not exchange_states:
            print(f'Completed {args.model_kind} signal in {(time.perf_counter()-t0)/60:.2f} min; augmented=0')
            return

        global_covered_dates = sorted({d for st in exchange_states.values() for d in st['covered_dates']})
        # Keep full-mode batches moderate to avoid OOM from wide in-memory Russell matrices.
        batch_size = 180 if not args.eval_times_only else 120
        n_batches = (len(global_covered_dates) + batch_size - 1) // batch_size

        for b in range(n_batches):
            batch_t0 = time.perf_counter()
            b0 = b * batch_size
            b1 = min(b0 + batch_size, len(global_covered_dates))
            batch_dates_global = global_covered_dates[b0:b1]
            if not batch_dates_global:
                continue
            batch_set_global = set(batch_dates_global)
            batch_start = pd.Timestamp(batch_dates_global[0])
            batch_end = pd.Timestamp(batch_dates_global[-1])

            # Build per-exchange worklists for this global date window.
            per_exchange = {}
            union_feature_tickers = set()
            needed_index_by_date = {} if args.eval_times_only else None
            for ex, st in exchange_states.items():
                ex_dates = [d for d in batch_dates_global if d in st['covered_set']]
                if not ex_dates:
                    continue
                feature_tickers = set()
                for ticker in st['tickers']:
                    for m in models_by_ticker.get(ticker, []):
                        if m['test_end'] < batch_start or m['test_start'] > batch_end:
                            continue
                        feature_tickers.update(m['feature_tickers'])
                ex_batch_tickers = [t for t in canonical_tickers if t in feature_tickers]
                if not ex_batch_tickers:
                    continue
                per_exchange[ex] = {
                    'dates': ex_dates,
                    'tickers': ex_batch_tickers,
                    'global_pos': np.array([global_ticker_pos[t] for t in ex_batch_tickers], dtype=np.int32),
                }
                union_feature_tickers.update(ex_batch_tickers)

                if args.eval_times_only:
                    # Include all timestamps needed by any exchange for each date in this batch.
                    for d in ex_dates:
                        base_ts = (
                            pd.Timestamp(f"{d} 09:35:00", tz='America/New_York')
                            if ex in ASIA_EXCHANGES
                            else st['close_df'].loc[d] + st['offset']
                        )
                        eval_idx = pd.DatetimeIndex([pd.Timestamp(f"{d} {t}:00", tz='America/New_York') for t in sorted(eval_times)])
                        want = eval_idx.union(pd.DatetimeIndex([base_ts])).sort_values()
                        prev = needed_index_by_date.get(d)
                        needed_index_by_date[d] = want if prev is None else prev.union(want).sort_values()

            if not per_exchange or not union_feature_tickers:
                if args.log_batch_timing:
                    print(f'[shared] batch {b+1}/{n_batches}: no exchange work', flush=True)
                continue

            load_t0 = time.perf_counter()
            russell_df = load_russell_batch_matrix(
                russell_ohlcv_dir,
                [t for t in canonical_tickers if t in union_feature_tickers],
                batch_set_global,
                needed_index_by_date=needed_index_by_date,
            )
            load_sec = time.perf_counter() - load_t0
            if russell_df.empty:
                if args.log_batch_timing:
                    print(
                        f'[shared] batch {b+1}/{n_batches}: empty russell_df '
                        f'(dates={len(batch_dates_global)} tickers={len(union_feature_tickers)} load={load_sec:.2f}s)',
                        flush=True,
                    )
                continue

            compute_t0 = time.perf_counter()
            for ex, work in per_exchange.items():
                st = exchange_states[ex]
                ex_tickers = work['tickers']
                available_cols = [c for c in ex_tickers if c in russell_df.columns]
                if not available_cols:
                    continue
                ex_pos_map = {t: i for i, t in enumerate(ex_tickers)}
                ex_valid_pos = np.array([ex_pos_map[t] for t in available_cols], dtype=np.int32)
                ex_global_pos_valid = work['global_pos'][ex_valid_pos]
                ex_russell_df = russell_df[available_cols]

                for date_str in work['dates']:
                    date_ts = pd.Timestamp(date_str)
                    close_time = st['close_df'].loc[date_str] + st['offset']
                    us_open_time = pd.Timestamp(f"{date_str} 09:35:00", tz='America/New_York')

                    active = []
                    for ticker in st['tickers']:
                        model_entry = get_model_for_date(models_by_ticker.get(ticker, []), date_ts)
                        if model_entry is not None:
                            active.append((ticker, model_entry))
                    if not active:
                        continue

                    day_matrix = ex_russell_df.loc[date_str:date_str]
                    day_matrix_ffill = day_matrix.ffill()
                    if ex in ASIA_EXCHANGES:
                        base_slice = day_matrix_ffill[day_matrix_ffill.index >= us_open_time]
                    else:
                        base_slice = day_matrix_ffill[day_matrix_ffill.index >= close_time]
                    if base_slice.empty:
                        continue

                    russell_base = base_slice.iloc[0].to_numpy(dtype=np.float32)
                    valid_mask = ~np.isnan(russell_base)
                    if not valid_mask.any():
                        continue

                    if ex in ASIA_EXCHANGES:
                        if date_str not in st['fut_us_open'].index:
                            continue
                        fut_base_price = float(st['fut_us_open'].loc[date_str, 'fut_us_open'])
                    else:
                        if date_str not in st['fut_domestic_close'].index:
                            continue
                        fut_base_price = float(st['fut_domestic_close'].loc[date_str, 'fut_domestic_close'])
                    if np.isnan(fut_base_price) or fut_base_price == 0:
                        continue

                    beta_dates = st['russell_betas'].index[st['russell_betas'].index <= date_ts]
                    if len(beta_dates) == 0:
                        continue
                    beta_row = st['russell_betas'].loc[beta_dates[-1]].reindex(ex_tickers).fillna(0.0)
                    beta_vec = beta_row.to_numpy(dtype=np.float32)[ex_valid_pos]

                    if args.eval_times_only:
                        pred_index = pd.DatetimeIndex(
                            [pd.Timestamp(f"{date_str} {t}:00", tz='America/New_York') for t in sorted(eval_times)]
                        ).sort_values()
                    else:
                        pred_index = base_slice.index.sort_values()
                    if ex in ASIA_EXCHANGES:
                        pred_index = pred_index[pred_index >= us_open_time]
                    if len(pred_index) == 0:
                        continue

                    valid_cols = np.where(valid_mask)[0]
                    valid_global_cols = ex_global_pos_valid[valid_cols]
                    aligned_prices = base_slice.iloc[:, valid_cols].reindex(pred_index, method='ffill')
                    price_arr = aligned_prices.to_numpy(dtype=np.float32)
                    base_arr = russell_base[valid_cols]
                    returns_arr = (price_arr - base_arr) / base_arr

                    fut_arr = st['futures_close_series'].reindex(pred_index, method='ffill').to_numpy(dtype=np.float32)
                    fut_ret = (fut_arr - fut_base_price) / fut_base_price
                    residual_arr = returns_arr - fut_ret[:, None] * beta_vec[valid_cols][None, :]
                    residual_arr = np.nan_to_num(residual_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    x_full = np.zeros((len(pred_index), len(canonical_tickers)), dtype=np.float32)
                    x_full[:, valid_global_cols] = residual_arr

                    linear_active = [(t, m) for t, m in active if m['kind'] == 'linear']
                    rf_active = [(t, m) for t, m in active if m['kind'] != 'linear']
                    if linear_active:
                        ltickers = [a[0] for a in linear_active]
                        w_mat = np.column_stack([a[1]['w'][valid_global_cols] for a in linear_active]).astype(np.float32)
                        c_vec = np.array([a[1]['c'] for a in linear_active], dtype=np.float32)
                        pred_mat = residual_arr @ w_mat + c_vec
                        pred_df = pd.DataFrame(pred_mat, index=pred_index, columns=ltickers)
                        for ticker, _ in linear_active:
                            out_df = pd.DataFrame({'signal': pred_df[ticker].to_numpy(dtype=np.float32)}, index=pred_index)
                            out_df['date'] = date_str
                            st['ticker_pred'][ticker].append(out_df)
                    for ticker, mentry in rf_active:
                        X_pred = x_full[:, mentry['feature_pos_global']]
                        p = mentry['model'].predict(X_pred)
                        out_df = pd.DataFrame({'signal': np.asarray(p, dtype=np.float32)}, index=pred_index)
                        out_df['date'] = date_str
                        st['ticker_pred'][ticker].append(out_df)

            if args.log_batch_timing:
                print(
                    f'[shared] batch {b+1}/{n_batches}: dates={len(batch_dates_global)} '
                    f'union_tickers={len(union_feature_tickers)} rows={len(russell_df)} '
                    f'cols={len(russell_df.columns)} load={load_sec:.2f}s '
                    f'compute={(time.perf_counter()-compute_t0):.2f}s total={(time.perf_counter()-batch_t0):.2f}s',
                    flush=True,
                )
            gc.collect()

        # Persist per-ticker outputs once after all batches.
        for ex, st in exchange_states.items():
            for ticker in st['tickers']:
                parts = st['ticker_pred'][ticker]
                out_df = pd.concat(parts) if parts else pd.DataFrame(columns=['signal', 'date'])
                tdir = output_dir / f'ticker={ticker}'
                tdir.mkdir(parents=True, exist_ok=True)
                out_df.to_parquet(tdir / 'data.parquet')
                if not out_df.empty:
                    overall_augmented.add(ticker)

    else:
        for ex, tickers in exchange_items:
            result = _run_exchange(
                ex, tickers, args, exchange_close_times[ex], stock_to_frd, models_by_ticker, canonical_tickers,
                global_ticker_pos, model_start, model_end, russell_betas_dir, futures_dir,
                russell_ohlcv_dir, eval_times, show_progress=True,
            )
            for ticker, out_df in result.items():
                tdir = output_dir / f'ticker={ticker}'
                tdir.mkdir(parents=True, exist_ok=True)
                out_df.to_parquet(tdir / 'data.parquet')
                if not out_df.empty:
                    overall_augmented.add(ticker)

    print(f'Completed {args.model_kind} signal in {(time.perf_counter()-t0)/60:.2f} min; augmented={len(overall_augmented)}')


if __name__ == '__main__':
    main()
