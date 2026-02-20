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
}
ASIA_EXCHANGES = {"XTKS", "XASX"}


class _LegacyRidgeResidualModel:
    """Placeholder class for unpickling legacy ridge artifacts."""
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-kind', choices=['ridge', 'pcr', 'robust_pcr', 'elasticnet', 'pls', 'rf', 'rrr', 'huber'], required=True)
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
            if canonical_features is None:
                canonical_features = feature_names

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
        canonical_features = []

    canonical_pos = {fn: i for i, fn in enumerate(canonical_features)}
    n_features = len(canonical_features)

    for ticker, entries in models.items():
        for e in entries:
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

    return models, canonical_features, model_start, model_end


def get_model_for_date(entries, d):
    for e in entries:
        if e['test_start'] <= d <= e['test_end']:
            return e
    return None


def load_russell_batch_matrix(russell_ohlcv_dir, tickers, dates_set):
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
    return pd.DataFrame(data).sort_index()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    params = load_params()
    start_date = params['start_date']
    end_date = params['end_date']

    data_dir = Path('.')
    futures_dir = data_dir / 'data' / 'processed' / 'futures' / 'converted_minute_bars'
    russell_ohlcv_dir = data_dir / 'data' / 'raw' / 'russell1000' / 'ohlcv-1m'
    russell_betas_dir = data_dir / 'data' / 'processed' / 'russell1000' / 'russell_betas'
    baseline_signal_dir = data_dir / 'data' / 'processed' / 'futures_only_signal'

    output_dir.mkdir(parents=True, exist_ok=True)
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

    models_by_ticker, canonical_features, model_start, model_end = load_models(model_dir)
    canonical_tickers = [fn.replace('russell_', '', 1) for fn in canonical_features]

    eligible_by_exchange = {}
    modeled_tickers = sorted(models_by_ticker.keys())
    for t in modeled_tickers:
        ex = exchange_dict.get(t)
        if ex:
            eligible_by_exchange.setdefault(ex, []).append(t)

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

    for ex, tickers in sorted(eligible_by_exchange.items()):
        offset = TIME_FUTURES_AFTER_CLOSE[ex]
        close_df = exchange_close_times[ex]

        betas_path = russell_betas_dir / f'{ex}.parquet'
        if not betas_path.exists():
            continue
        russell_betas = pd.read_parquet(betas_path)
        russell_betas.index = pd.to_datetime(russell_betas.index)

        futures_symbol = None
        for t in tickers:
            fs = stock_to_frd.get(t)
            if fs:
                futures_symbol = fs
                break
        if futures_symbol is None:
            continue

        futures_df_raw = pd.read_parquet(
            futures_dir,
            filters=[('symbol', '==', futures_symbol), ('timestamp', '>=', pd.Timestamp(start_date, tz='America/New_York'))],
            columns=['timestamp', 'symbol', 'close'],
        ).sort_values('timestamp')
        futures_df_raw['date'] = futures_df_raw['timestamp'].dt.strftime('%Y-%m-%d')
        futures_df_raw = futures_df_raw.set_index('timestamp')

        merged_fut = futures_df_raw.merge(close_df.rename('domestic_close_time'), left_on='date', right_index=True)
        fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time', 'close']].apply(
            lambda x: x[x.index <= x['domestic_close_time'] + offset].iloc[-1]['close']
            if (x.index <= x['domestic_close_time'] + offset).any() else np.nan
        ).to_frame(name='fut_domestic_close')
        futures_close_series = futures_df_raw['close'].astype(np.float32)

        baseline_by_date = {}
        for ticker in tickers:
            bp = baseline_signal_dir / f'ticker={ticker}' / 'data.parquet'
            if bp.exists():
                bdf = pd.read_parquet(bp)
                baseline_by_date[ticker] = {d: g[['signal', 'date']] for d, g in bdf.groupby('date', sort=False)}

        ticker_augmented = {t: [] for t in tickers}
        available_dates = sorted(close_df.index.tolist())
        covered_dates = [d for d in available_dates if model_start <= pd.Timestamp(d) <= model_end]

        batch_size = 120
        n_batches = (len(covered_dates) + batch_size - 1) // batch_size

        for b in range(n_batches):
            b0 = b * batch_size
            b1 = min(b0 + batch_size, len(covered_dates))
            batch_dates = covered_dates[b0:b1]
            batch_set = set(batch_dates)

            russell_df = load_russell_batch_matrix(russell_ohlcv_dir, canonical_tickers, batch_set)
            if russell_df.empty:
                continue

            for date_str in tqdm(batch_dates, desc=f'{ex} {args.model_kind} {b+1}/{n_batches}'):
                date_ts = pd.Timestamp(date_str)
                close_time = close_df.loc[date_str] + offset
                us_open_time = pd.Timestamp(f"{date_str} 09:30:00", tz="America/New_York")

                active = []
                for ticker in tickers:
                    day_df = baseline_by_date.get(ticker, {}).get(date_str)
                    if day_df is None or day_df.empty:
                        continue
                    model_entry = get_model_for_date(models_by_ticker.get(ticker, []), date_ts)
                    if model_entry is None:
                        continue
                    active.append((ticker, model_entry, day_df))

                if not active:
                    continue

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

                if date_str not in fut_domestic_close.index:
                    continue

                fut_close_price = float(fut_domestic_close.loc[date_str, 'fut_domestic_close'])
                if np.isnan(fut_close_price) or fut_close_price == 0:
                    continue

                beta_dates = russell_betas.index[russell_betas.index <= date_ts]
                if len(beta_dates) == 0:
                    continue
                beta_row = russell_betas.loc[beta_dates[-1]].reindex(canonical_tickers).fillna(0.0)
                beta_vec = beta_row.to_numpy(dtype=np.float32)

                pred_index = active[0][2].index
                for _, _, bd in active[1:]:
                    if not bd.index.equals(pred_index):
                        pred_index = pred_index.union(bd.index)
                pred_index = pred_index.sort_values()

                valid_cols = np.where(valid_mask)[0]
                aligned_prices = base_slice.iloc[:, valid_cols].reindex(pred_index, method='ffill')
                price_arr = aligned_prices.to_numpy(dtype=np.float32)
                base_arr = russell_base[valid_cols]
                returns_arr = (price_arr - base_arr) / base_arr

                fut_arr = futures_close_series.reindex(pred_index, method='ffill').to_numpy(dtype=np.float32)
                fut_ret = (fut_arr - fut_close_price) / fut_close_price

                residual_arr = returns_arr - fut_ret[:, None] * beta_vec[valid_cols][None, :]
                residual_arr = np.nan_to_num(residual_arr, nan=0.0, posinf=0.0, neginf=0.0)

                x_full = np.zeros((len(pred_index), len(canonical_tickers)), dtype=np.float32)
                x_full[:, valid_cols] = residual_arr

                # linear models can be batched
                linear_active = [(t, m, d) for t, m, d in active if m['kind'] == 'linear']
                rf_active = [(t, m, d) for t, m, d in active if m['kind'] != 'linear']

                if linear_active:
                    ltickers = [a[0] for a in linear_active]
                    w_mat = np.column_stack([a[1]['w'] for a in linear_active]).astype(np.float32)
                    c_vec = np.array([a[1]['c'] for a in linear_active], dtype=np.float32)
                    pred_mat = x_full @ w_mat + c_vec
                    pred_df = pd.DataFrame(pred_mat, index=pred_index, columns=ltickers)
                    for ticker, _, baseline_date_df in linear_active:
                        p = pred_df[ticker].reindex(baseline_date_df.index, method='ffill').to_numpy(dtype=np.float32)
                        aug = baseline_date_df[['signal']].copy()
                        aug['signal'] = aug['signal'].to_numpy(dtype=np.float32) + p
                        aug['date'] = date_str
                        ticker_augmented[ticker].append(aug)

                for ticker, mentry, baseline_date_df in rf_active:
                    # reindex to model features then predict
                    feature_names = mentry['feature_names']
                    feat_tickers = [f.replace('russell_', '', 1) for f in feature_names]
                    feat_pos = [canonical_tickers.index(t) for t in feat_tickers if t in canonical_tickers]
                    X_pred = x_full[:, feat_pos]
                    p = mentry['model'].predict(X_pred)
                    ps = pd.Series(p, index=pred_index).reindex(baseline_date_df.index, method='ffill').to_numpy(dtype=np.float32)
                    aug = baseline_date_df[['signal']].copy()
                    aug['signal'] = aug['signal'].to_numpy(dtype=np.float32) + ps
                    aug['date'] = date_str
                    ticker_augmented[ticker].append(aug)

            gc.collect()

        for ticker in tickers:
            parts = ticker_augmented[ticker]
            out_df = pd.concat(parts) if parts else pd.DataFrame(columns=['signal', 'date'])

            tdir = output_dir / f'ticker={ticker}'
            tdir.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(tdir / 'data.parquet')
            if parts:
                overall_augmented.add(ticker)

        del futures_df_raw, merged_fut, fut_domestic_close, futures_close_series, russell_betas
        gc.collect()

    print(f'Completed {args.model_kind} signal in {(time.perf_counter()-t0)/60:.2f} min; augmented={len(overall_augmented)}')


if __name__ == '__main__':
    main()
