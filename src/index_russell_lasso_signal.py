"""
Generate LASSO-augmented intraday signal for ADR arbitrage.

For LASSO-eligible ADRs (IC improvement >= 0.04):
  signal = baseline_signal + lasso_residual_pred

For non-LASSO ADRs:
  Copies baseline signal from data/processed/futures_only_signal/

LASSO residual prediction uses Russell 1000 intraday returns, residualized
against the index futures return, then fed through the trained LASSO model.

Memory-efficient: only loads Russell tickers with non-zero LASSO coefficients.
Uses sparse prediction to bypass full scaler.transform().
"""

import os
import sys
import pickle
import shutil
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from utils import load_params
from utils_lasso_residuals import load_index_mapping, INDEX_FUTURE_TO_SYMBOL
from train_lasso_models import LASSOResidualModel

__script_dir__ = Path(__file__).parent.absolute()

# LASSO-eligible tickers (mean IC improvement >= 0.04)
LASSO_ELIGIBLE = [
    'ARGX', 'ASML', 'AZN', 'BP', 'BTI', 'BUD', 'DEO', 'E', 'EQNR',
    'FMS', 'GMAB', 'GSK', 'HLN', 'IHG', 'NGG', 'NVS', 'RIO',
    'SHEL', 'SNY', 'TS', 'TTE', 'UL',
]

# Offset between exchange close and the actual closing auction time
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


def load_lasso_models(model_dir):
    """
    Load all LASSO models for eligible tickers.

    Returns:
        dict: {ticker: [(test_start, test_end, model_data), ...]}
    """
    models = {}
    for ticker in LASSO_ELIGIBLE:
        ticker_dir = model_dir / ticker
        if not ticker_dir.exists():
            print(f"  Warning: No models for {ticker}")
            continue

        model_files = sorted(ticker_dir.glob('*.pkl'))
        ticker_models = []
        for mf in model_files:
            with open(mf, 'rb') as f:
                model_data = pickle.load(f)
            test_start, test_end = model_data['test_period']
            ticker_models.append((test_start, test_end, model_data))

        models[ticker] = ticker_models

    return models


def precompute_sparse_models(models):
    """
    Precompute adjusted coefficients for sparse prediction.

    Since fit_intercept=False (intercept=0):
      prediction = sum_{i in nz} (coef[i]/scale[i]) * feature[i]
                   - sum_{i in nz} (coef[i]*mean[i]/scale[i])

    Returns:
        sparse_models: {ticker: [(test_start, test_end, sparse_data), ...]}
    """
    sparse_models = {}

    for ticker, ticker_models in models.items():
        sparse_ticker = []
        for test_start, test_end, model_data in ticker_models:
            model_obj = model_data['model']
            feature_names = model_data['feature_names']
            coefs = model_obj.model.coef_
            nonzero_idx = np.where(coefs != 0)[0]

            if len(nonzero_idx) == 0:
                sparse_ticker.append((test_start, test_end, {
                    'offset': 0.0,
                    'nonzero_tickers': [],
                    'adjusted_coefs': np.array([]),
                }))
            else:
                nz_coefs = coefs[nonzero_idx]
                nz_means = model_obj.scaler.mean_[nonzero_idx]
                nz_scales = model_obj.scaler.scale_[nonzero_idx]

                adjusted_coefs = nz_coefs / nz_scales
                offset = -np.sum(nz_coefs * nz_means / nz_scales)

                nz_tickers = [feature_names[i].replace('russell_', '', 1)
                              for i in nonzero_idx]

                sparse_ticker.append((test_start, test_end, {
                    'offset': offset,
                    'nonzero_tickers': nz_tickers,
                    'adjusted_coefs': adjusted_coefs,
                }))

        sparse_models[ticker] = sparse_ticker

    return sparse_models


def get_needed_tickers_for_exchange(sparse_models, eligible_tickers):
    """Get union of non-zero Russell tickers needed for one exchange's models."""
    needed = set()
    for ticker in eligible_tickers:
        if ticker not in sparse_models:
            continue
        for _, _, sparse_data in sparse_models[ticker]:
            needed.update(sparse_data['nonzero_tickers'])
    return needed


def get_sparse_model_for_date(sparse_ticker_models, date_ts):
    """Find the sparse model whose test_period contains the given date."""
    for test_start, test_end, sparse_data in sparse_ticker_models:
        if test_start <= date_ts <= test_end:
            return sparse_data
    return None


def load_russell_minute_data(russell_ohlcv_dir, tickers, dates_set):
    """
    Load Russell minute bar close prices for specified tickers only.
    No close-time filtering — data shared across exchanges.

    Returns:
        dict: {ticker: Series with tz-aware ET index and float32 Close values}
    """
    all_series = {}
    for ticker in tqdm(sorted(tickers), desc="Loading Russell minute bars"):
        parquet_path = russell_ohlcv_dir / f'ticker={ticker}' / 'data.parquet'
        if not parquet_path.exists():
            continue

        try:
            df = pd.read_parquet(parquet_path, columns=['Close', 'date'])
            df = df[df['date'].isin(dates_set)]
            if df.empty:
                continue
            df.index = df.index.tz_localize('America/New_York')
            all_series[ticker] = df['Close'].astype(np.float32)
        except Exception:
            continue

    print(f"  Loaded {len(all_series)} Russell tickers")
    return all_series


def main():
    print("=" * 70)
    print("LASSO-Augmented Intraday Signal Generation")
    print("=" * 70)

    params = load_params()
    start_date = params['start_date']
    end_date = params['end_date']
    print(f"Date range: {start_date} to {end_date}")

    # Paths
    data_dir = __script_dir__ / '..'
    futures_dir = data_dir / 'data' / 'processed' / 'futures' / 'converted_minute_bars'
    russell_ohlcv_dir = data_dir / 'data' / 'raw' / 'russell1000' / 'ohlcv-1m'
    russell_betas_dir = data_dir / 'data' / 'processed' / 'russell1000' / 'russell_betas'
    model_dir = data_dir / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'lasso'
    baseline_signal_dir = data_dir / 'data' / 'processed' / 'futures_only_signal'
    output_dir = data_dir / 'data' / 'processed' / 'index_russell_lasso_signal'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ADR info
    adr_info_path = data_dir / 'data' / 'raw' / 'adr_info.csv'
    adr_info = pd.read_csv(adr_info_path)
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity', '')
    adr_tickers = adr_info['adr'].tolist()
    exchange_dict = adr_info.set_index('adr')['exchange'].to_dict()

    # Load futures symbols
    futures_symbols_path = data_dir / 'data' / 'raw' / 'futures_symbols.csv'
    futures_symbols = pd.read_csv(futures_symbols_path)

    # Map ADR tickers to FRD futures symbols
    adr_info['index_future_bbg'] = adr_info['index_future_bbg'].str.strip()
    futures_symbols['bloomberg_symbol'] = futures_symbols['bloomberg_symbol'].str.strip()
    merged_info = adr_info.merge(futures_symbols, left_on='index_future_bbg',
                                  right_on='bloomberg_symbol')
    stock_to_frd = merged_info.set_index('adr_x' if 'adr_x' in merged_info.columns else 'adr')[
        'first_rate_symbol'].to_dict()

    # Load LASSO models and precompute sparse prediction params
    print("\nLoading LASSO models...")
    models = load_lasso_models(model_dir)
    print(f"Loaded models for {len(models)} tickers")

    print("Precomputing sparse model parameters...")
    sparse_models = precompute_sparse_models(models)
    del models  # free raw model objects

    # Group eligible tickers by exchange
    eligible_by_exchange = {}
    for ticker in LASSO_ELIGIBLE:
        ex = exchange_dict.get(ticker)
        if ex:
            eligible_by_exchange.setdefault(ex, []).append(ticker)

    print(f"\nLASSO-eligible tickers by exchange:")
    for ex, tickers in eligible_by_exchange.items():
        needed = get_needed_tickers_for_exchange(sparse_models, tickers)
        print(f"  {ex}: {tickers} ({len(needed)} Russell tickers needed)")

    # Process each exchange (load Russell data per-exchange to limit memory)
    all_results = {}
    for exchange_mic, eligible_tickers in eligible_by_exchange.items():
        print(f"\n{'=' * 60}")
        print(f"Processing exchange: {exchange_mic}")
        print(f"{'=' * 60}")

        offset = TIME_FUTURES_AFTER_CLOSE[exchange_mic]

        # Close times for this exchange
        close_df = (
            mcal.get_calendar(exchange_mic)
            .schedule(start_date=start_date, end_date=end_date)['market_close']
            .dt.tz_convert('America/New_York')
        ).rename('domestic_close_time')
        close_df.index = close_df.index.astype(str)
        available_dates = sorted(close_df.index.tolist())

        # Get needed Russell tickers for this exchange only
        needed_tickers = get_needed_tickers_for_exchange(
            sparse_models, eligible_tickers)
        print(f"  Need {len(needed_tickers)} Russell tickers for this exchange")

        if not needed_tickers:
            print(f"  No non-zero models, skipping")
            continue

        # Load Russell betas
        betas_path = russell_betas_dir / f'{exchange_mic}.parquet'
        if not betas_path.exists():
            print(f"  Russell betas not found, skipping")
            continue
        russell_betas = pd.read_parquet(betas_path)
        russell_betas.index = pd.to_datetime(russell_betas.index)

        # Load Russell minute data for THIS exchange's needed tickers
        exchange_dates = set(available_dates)
        russell_data = load_russell_minute_data(
            russell_ohlcv_dir, needed_tickers, exchange_dates)

        # Load futures data for this exchange
        futures_symbol = None
        for ticker in eligible_tickers:
            fs = stock_to_frd.get(ticker)
            if fs:
                futures_symbol = fs
                break
        if futures_symbol is None:
            print(f"  No futures mapping, skipping")
            del russell_data
            continue

        print(f"  Loading futures data for {futures_symbol}...")
        futures_df_raw = pd.read_parquet(
            futures_dir,
            filters=[('timestamp', '>=', pd.Timestamp(start_date, tz='America/New_York'))],
            columns=['timestamp', 'symbol', 'close']
        )
        futures_df_raw = futures_df_raw[futures_df_raw['symbol'] == futures_symbol].copy()
        futures_df_raw['date'] = futures_df_raw['timestamp'].dt.strftime('%Y-%m-%d')
        futures_df_raw = futures_df_raw.set_index('timestamp')

        # Futures close price at exchange close for each date
        merged_fut = futures_df_raw.merge(
            close_df.rename('domestic_close_time'), left_on='date', right_index=True)
        fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time', 'close']].apply(
            lambda x: x[x.index <= x['domestic_close_time'] + offset].iloc[-1]['close']
            if (x.index <= x['domestic_close_time'] + offset).any() else np.nan
        ).to_frame(name='fut_domestic_close')

        futures_close_series = futures_df_raw['close']

        # Load baseline signals for all eligible tickers
        baseline_signals = {}
        for ticker in eligible_tickers:
            bp = baseline_signal_dir / f'ticker={ticker}' / 'data.parquet'
            if bp.exists():
                baseline_signals[ticker] = pd.read_parquet(bp)

        # Results storage per ticker
        ticker_augmented = {t: [] for t in eligible_tickers}
        ticker_fallback_count = {t: 0 for t in eligible_tickers}

        # Process date by date
        print(f"  Processing {len(available_dates)} dates...")
        for date_str in tqdm(available_dates, desc=f"  {exchange_mic}"):
            date_ts = pd.Timestamp(date_str)
            close_time = close_df.loc[date_str] + offset

            # Get Russell data for this date, after close
            # Build small DataFrame on the fly from dict of Series
            day_cols = {}
            for t, series in russell_data.items():
                day_series = series.loc[date_str:date_str]
                after = day_series[day_series.index >= close_time]
                if not after.empty:
                    day_cols[t] = after
            if not day_cols:
                for t in eligible_tickers:
                    ticker_fallback_count[t] += 1
                continue

            after_close = pd.DataFrame(day_cols)
            # Forward-fill within this date
            after_close = after_close.ffill()

            # Russell close prices (first row at/after close time)
            russell_close = after_close.iloc[0]
            valid_tickers = russell_close.dropna().index

            # Check futures
            if date_str not in fut_domestic_close.index:
                for t in eligible_tickers:
                    ticker_fallback_count[t] += 1
                continue
            fut_close_price = fut_domestic_close.loc[date_str, 'fut_domestic_close']
            if np.isnan(fut_close_price):
                for t in eligible_tickers:
                    ticker_fallback_count[t] += 1
                continue

            # Get Russell betas for this date
            beta_dates = russell_betas.index[russell_betas.index <= date_ts]
            if len(beta_dates) == 0:
                for t in eligible_tickers:
                    ticker_fallback_count[t] += 1
                continue
            date_betas = russell_betas.loc[beta_dates[-1]]
            betas_aligned = date_betas.reindex(valid_tickers).fillna(0.0)

            # Process each eligible ticker on this exchange
            for ticker in eligible_tickers:
                if ticker not in baseline_signals:
                    continue
                if ticker not in sparse_models:
                    ticker_fallback_count[ticker] += 1
                    continue

                # Get sparse model for this date
                sparse_data = get_sparse_model_for_date(
                    sparse_models[ticker], date_ts)
                if sparse_data is None:
                    ticker_fallback_count[ticker] += 1
                    continue

                nz_tickers = sparse_data['nonzero_tickers']
                if not nz_tickers:
                    ticker_fallback_count[ticker] += 1
                    continue

                # Get baseline timestamps for this date
                baseline_df = baseline_signals[ticker]
                baseline_date = baseline_df[baseline_df['date'] == date_str]
                if baseline_date.empty:
                    continue

                # Align Russell after-close data to baseline timestamps
                aligned_russell = after_close[valid_tickers].reindex(
                    baseline_date.index, method='ffill')

                # Compute Russell returns since close
                returns_matrix = (aligned_russell - russell_close[valid_tickers]) / russell_close[valid_tickers]

                # Compute futures returns at each baseline timestamp
                fut_prices = futures_close_series.reindex(
                    baseline_date.index, method='ffill')
                index_fut_returns = (fut_prices - fut_close_price) / fut_close_price

                # Residualize: residual = return - beta * index_return
                residuals_matrix = returns_matrix.sub(
                    index_fut_returns.values[:, np.newaxis] * betas_aligned.values[np.newaxis, :],
                )
                residuals_matrix = residuals_matrix.fillna(0.0)

                # Sparse prediction: only use non-zero coefficient tickers
                available_nz = [t for t in nz_tickers
                                if t in residuals_matrix.columns]
                if not available_nz:
                    ticker_fallback_count[ticker] += 1
                    continue

                feature_vals = residuals_matrix[available_nz].values
                coef_map = dict(zip(nz_tickers, sparse_data['adjusted_coefs']))
                coefs = np.array([coef_map[t] for t in available_nz])

                preds = sparse_data['offset'] + feature_vals @ coefs

                # Add LASSO prediction to baseline signal
                augmented = baseline_date[['signal']].copy()
                augmented['signal'] = augmented['signal'].values + preds
                augmented['date'] = date_str
                ticker_augmented[ticker].append(augmented)

        # Free Russell data for this exchange
        del russell_data

        # Combine results for this exchange
        for ticker in eligible_tickers:
            parts_list = ticker_augmented[ticker]
            if parts_list:
                augmented_signal = pd.concat(parts_list)
            else:
                augmented_signal = pd.DataFrame(columns=['signal', 'date'])

            # Fill missing dates from baseline
            if ticker in baseline_signals:
                baseline_df = baseline_signals[ticker]
                all_baseline_dates = set(baseline_df['date'].unique())
                augmented_dates = set(augmented_signal['date'].unique()) if not augmented_signal.empty else set()
                missing_dates = all_baseline_dates - augmented_dates
                if missing_dates:
                    fallback = baseline_df[baseline_df['date'].isin(missing_dates)]
                    augmented_signal = pd.concat([augmented_signal, fallback]).sort_index()

            all_results[ticker] = augmented_signal
            n_aug = len(parts_list)
            n_fb = ticker_fallback_count[ticker]
            print(f"  {ticker}: {n_aug} dates augmented, {n_fb} fallback")

    # Save augmented signals
    print(f"\n{'=' * 60}")
    print("Saving results...")
    for ticker, signal_df in all_results.items():
        ticker_output = output_dir / f'ticker={ticker}'
        ticker_output.mkdir(parents=True, exist_ok=True)
        signal_df.to_parquet(ticker_output / 'data.parquet')
        print(f"  Saved augmented signal for {ticker}: {len(signal_df)} rows")

    # Copy non-LASSO tickers from baseline
    non_lasso_tickers = [t for t in adr_tickers if t not in LASSO_ELIGIBLE]
    print(f"\nCopying baseline signal for {len(non_lasso_tickers)} non-LASSO tickers...")
    for ticker in non_lasso_tickers:
        src_path = baseline_signal_dir / f'ticker={ticker}'
        dst_path = output_dir / f'ticker={ticker}'
        if not src_path.exists():
            continue
        dst_path.mkdir(parents=True, exist_ok=True)
        src_file = src_path / 'data.parquet'
        dst_file = dst_path / 'data.parquet'
        if src_file.exists():
            shutil.copy2(src_file, dst_file)

    # Copy LASSO-eligible tickers that had no models
    for ticker in LASSO_ELIGIBLE:
        if ticker not in all_results:
            src_path = baseline_signal_dir / f'ticker={ticker}'
            dst_path = output_dir / f'ticker={ticker}'
            if src_path.exists() and not dst_path.exists():
                dst_path.mkdir(parents=True, exist_ok=True)
                src_file = src_path / 'data.parquet'
                if src_file.exists():
                    shutil.copy2(src_file, dst_path / 'data.parquet')
                    print(f"  Copied baseline for {ticker} (no models available)")

    print(f"\n{'=' * 70}")
    print(f"Signal generation complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  LASSO-augmented: {len(all_results)} tickers")
    print(f"  Baseline copied: {len(non_lasso_tickers)} tickers")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
