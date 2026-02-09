"""
Generate LASSO-augmented intraday signal for ADR arbitrage.

For LASSO-eligible ADRs (IC improvement >= 0.04):
  signal = beta * futures_return_since_close + lasso_residual_pred - adr_return_since_close

For non-LASSO ADRs:
  Copies baseline signal from data/processed/futures_only_signal/

LASSO residual prediction uses Russell 1000 intraday returns, residualized
against the index futures return, then fed through the trained LASSO model.
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
    'ARGX', 'ASML', 'AZN', 'BP', 'BTI', 'DEO', 'E', 'EQNR',
    'GSK', 'HLN', 'IHG', 'NGG', 'NVS', 'RIO', 'SHEL', 'TS', 'TTE', 'UL',
]

# Offset between exchange close and the actual closing auction time
# (same values used in only_futures_full_signal.py)
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
    Load all LASSO models for eligible tickers, organized by ticker and date range.

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


def get_model_for_date(ticker_models, date):
    """
    Find the model whose test_period contains the given date.

    Returns model_data dict or None.
    """
    date_ts = pd.Timestamp(date)
    for test_start, test_end, model_data in ticker_models:
        if test_start <= date_ts <= test_end:
            return model_data
    return None


def build_russell_wide_df(russell_ohlcv_dir, tickers, dates_set, close_times_by_date):
    """
    Load Russell minute bars and build a wide DataFrame: (timestamp x ticker) of Close prices.
    Only includes timestamps at or after exchange close for each date.

    Returns:
        DataFrame with tz-aware ET DatetimeIndex, columns = ticker names, values = Close price
    """
    all_series = {}
    loaded = 0

    for ticker in tqdm(tickers, desc="Loading Russell minute bars"):
        parquet_path = russell_ohlcv_dir / f'ticker={ticker}' / 'data.parquet'
        if not parquet_path.exists():
            continue

        try:
            df = pd.read_parquet(parquet_path, columns=['Close', 'date'])
            df = df[df['date'].isin(dates_set)]
            if df.empty:
                continue

            df.index = df.index.tz_localize('America/New_York')

            # Filter to timestamps at or after exchange close
            keep_mask = pd.Series(False, index=df.index)
            for date_str, close_time in close_times_by_date.items():
                day_mask = df['date'] == date_str
                if day_mask.any():
                    keep_mask = keep_mask | (day_mask & (df.index >= close_time))

            filtered = df[keep_mask]['Close']
            if not filtered.empty:
                all_series[ticker] = filtered
                loaded += 1
        except Exception:
            continue

    print(f"  Loaded minute bars for {loaded} Russell tickers")
    if not all_series:
        return pd.DataFrame()

    # Combine into wide DataFrame (timestamps as index, tickers as columns)
    wide = pd.DataFrame(all_series)
    return wide


def compute_lasso_signal_for_exchange(
    exchange_mic,
    eligible_tickers_on_exchange,
    models,
    russell_ohlcv_dir,
    russell_betas_path,
    futures_dir,
    adr_info,
    futures_symbols,
    close_times_df,
    start_date,
    end_date,
):
    """
    Compute LASSO-augmented signal for all eligible tickers on one exchange.

    Vectorized: builds a wide Russell price DataFrame, computes returns and
    residuals as matrix operations, then applies LASSO models.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing exchange: {exchange_mic}")
    print(f"Eligible tickers: {eligible_tickers_on_exchange}")
    print(f"{'=' * 60}")

    offset = TIME_FUTURES_AFTER_CLOSE[exchange_mic]

    # Get exchange-to-index mapping
    _, exchange_to_index = load_index_mapping()
    index_symbol = exchange_to_index.get(exchange_mic)
    if index_symbol is None:
        print(f"  No index mapping for {exchange_mic}, skipping")
        return {}

    # Load Russell betas for this exchange
    if not russell_betas_path.exists():
        print(f"  Russell betas not found: {russell_betas_path}, skipping")
        return {}
    russell_betas = pd.read_parquet(russell_betas_path)
    russell_betas.index = pd.to_datetime(russell_betas.index)
    russell_tickers = russell_betas.columns.tolist()
    print(f"  Russell betas: {russell_betas.shape}")

    # Close times for this exchange
    close_df = close_times_df
    available_dates = sorted(close_df.index.tolist())

    # Build close_times_by_date
    close_times_by_date = {}
    for date_str in available_dates:
        close_times_by_date[date_str] = close_df.loc[date_str] + offset

    # Build wide Russell price DataFrame (vectorized approach)
    dates_set = set(available_dates)
    print(f"  Loading Russell minute bars for {len(russell_tickers)} tickers, {len(available_dates)} dates...")
    russell_wide = build_russell_wide_df(russell_ohlcv_dir, russell_tickers, dates_set, close_times_by_date)

    if russell_wide.empty:
        print(f"  No Russell minute data available, skipping exchange")
        return {}

    # Forward-fill within each date to handle missing minutes
    date_labels = russell_wide.index.normalize()
    parts = []
    for day_ts in date_labels.unique():
        day_slice = russell_wide[date_labels == day_ts].ffill()
        parts.append(day_slice)
    russell_wide = pd.concat(parts)

    # Compute Russell close price at exchange close for each date
    # (first minute at or after close time for each date)
    print(f"  Computing Russell close prices at exchange close...")
    russell_close_prices = {}
    for date_str in available_dates:
        close_time = close_times_by_date[date_str]
        day_ts = pd.Timestamp(date_str, tz='America/New_York')
        day_data = russell_wide[russell_wide.index.normalize() == day_ts]
        at_close = day_data[day_data.index >= close_time]
        if not at_close.empty:
            russell_close_prices[date_str] = at_close.iloc[0]

    # Map ADR tickers to futures symbols
    merged_info = adr_info.merge(futures_symbols, left_on='index_future_bbg', right_on='bloomberg_symbol')
    stock_to_frd = merged_info.set_index(merged_info['adr'].str.replace(' US Equity', ''))['first_rate_symbol'].to_dict()

    # Load futures data once for this exchange's index future
    # All tickers on same exchange use the same index future
    # Get the futures symbol from the first eligible ticker
    futures_symbol = None
    for ticker in eligible_tickers_on_exchange:
        fs = stock_to_frd.get(ticker)
        if fs:
            futures_symbol = fs
            break

    if futures_symbol is None:
        print(f"  No futures mapping for any ticker on {exchange_mic}")
        return {}

    print(f"  Loading futures data for {futures_symbol}...")
    futures_df_raw = pd.read_parquet(
        futures_dir,
        filters=[('timestamp', '>=', pd.Timestamp(start_date, tz='America/New_York'))],
        columns=['timestamp', 'symbol', 'close']
    )
    futures_df_raw = futures_df_raw[futures_df_raw['symbol'] == futures_symbol].copy()
    futures_df_raw['date'] = futures_df_raw['timestamp'].dt.strftime('%Y-%m-%d')
    futures_df_raw = futures_df_raw.set_index('timestamp')

    # Get futures close price at exchange close for each date
    merged_fut = futures_df_raw.merge(close_df.rename('domestic_close_time'), left_on='date', right_index=True)
    fut_domestic_close = merged_fut.groupby('date')[['domestic_close_time', 'close']].apply(
        lambda x: x[x.index <= x['domestic_close_time'] + offset].iloc[-1]['close']
        if (x.index <= x['domestic_close_time'] + offset).any() else np.nan
    ).to_frame(name='fut_domestic_close')

    # Reindex futures for fast lookup: forward-fill so every minute has a price
    futures_close_series = futures_df_raw['close']

    # Precompute LASSO predictions for each (date, minute) — shared across all tickers on this exchange
    # that use the same model feature set. However, each ticker may use a different model.
    # So we compute predictions per ticker.

    results = {}
    for ticker in eligible_tickers_on_exchange:
        print(f"\n  Processing {ticker}...")

        # Load baseline signal
        baseline_path = __script_dir__ / '..' / 'data' / 'processed' / 'futures_only_signal' / f'ticker={ticker}' / 'data.parquet'
        if not baseline_path.exists():
            print(f"    No baseline signal, skipping")
            continue
        baseline_signal = pd.read_parquet(baseline_path)

        # Get this ticker's LASSO models
        ticker_models = models.get(ticker)
        if not ticker_models:
            print(f"    No LASSO models, copying baseline")
            results[ticker] = baseline_signal
            continue

        augmented_parts = []
        dates_processed = 0
        dates_fallback = 0

        for date_str in available_dates:
            date_ts = pd.Timestamp(date_str)

            # Get model for this date
            model_data = get_model_for_date(ticker_models, date_ts)
            if model_data is None:
                dates_fallback += 1
                continue

            model = model_data['model']
            feature_names = model_data['feature_names']

            # Check Russell close prices exist for this date
            if date_str not in russell_close_prices:
                dates_fallback += 1
                continue

            russ_close = russell_close_prices[date_str]

            # Check futures close price at exchange close
            if date_str not in fut_domestic_close.index:
                dates_fallback += 1
                continue
            fut_close_price = fut_domestic_close.loc[date_str, 'fut_domestic_close']
            if np.isnan(fut_close_price):
                dates_fallback += 1
                continue

            # Get Russell betas for this date (most recent available)
            beta_dates = russell_betas.index[russell_betas.index <= date_ts]
            if len(beta_dates) == 0:
                dates_fallback += 1
                continue
            date_betas = russell_betas.loc[beta_dates[-1]]

            # Get baseline signal timestamps for this date
            baseline_date = baseline_signal[baseline_signal['date'] == date_str]
            if baseline_date.empty:
                continue

            # Get Russell prices for all minutes in this date (vectorized)
            day_ts_tz = pd.Timestamp(date_str, tz='America/New_York')
            day_russell = russell_wide[russell_wide.index.normalize() == day_ts_tz]
            if day_russell.empty:
                dates_fallback += 1
                continue

            # Align Russell data to baseline signal timestamps
            # Use reindex with ffill to get Russell prices at each baseline timestamp
            common_cols = day_russell.columns
            aligned_russell = day_russell.reindex(baseline_date.index, method='ffill')

            # Compute returns: (current_price - close_price) / close_price
            # russ_close is a Series indexed by ticker
            russ_close_aligned = russ_close.reindex(common_cols)
            valid_cols = russ_close_aligned.dropna().index
            returns_matrix = (aligned_russell[valid_cols] - russ_close_aligned[valid_cols]) / russ_close_aligned[valid_cols]

            # Get futures prices at each baseline timestamp (for residualization)
            fut_prices = futures_close_series.reindex(baseline_date.index, method='ffill')
            index_fut_returns = (fut_prices - fut_close_price) / fut_close_price

            # Residualize: russell_residual = russell_return - beta * index_fut_return
            betas_aligned = date_betas.reindex(valid_cols).fillna(0.0)
            residuals_matrix = returns_matrix.sub(
                index_fut_returns.values[:, np.newaxis] * betas_aligned.values[np.newaxis, :],
            )
            residuals_matrix = residuals_matrix.fillna(0.0)

            # Build feature matrix matching model's feature_names order
            # feature_names are like 'russell_AAPL' -> map to ticker 'AAPL'
            russ_ticker_order = [fn.replace('russell_', '', 1) for fn in feature_names]

            # Reindex residuals to match feature order, fill missing with 0
            feature_matrix = residuals_matrix.reindex(columns=russ_ticker_order, fill_value=0.0).values

            # Apply LASSO model (vectorized prediction for all minutes at once)
            feature_scaled = model.scaler.transform(feature_matrix)
            lasso_preds = model.model.predict(feature_scaled)

            # Add LASSO prediction to baseline signal
            augmented = baseline_date[['signal']].copy()
            augmented['signal'] = augmented['signal'].values + lasso_preds
            augmented['date'] = date_str
            augmented_parts.append(augmented)
            dates_processed += 1

        # Combine augmented dates
        if augmented_parts:
            augmented_signal = pd.concat(augmented_parts)
        else:
            augmented_signal = pd.DataFrame(columns=['signal', 'date'])

        # For dates without LASSO coverage, use baseline
        all_baseline_dates = set(baseline_signal['date'].unique())
        augmented_dates = set(augmented_signal['date'].unique()) if not augmented_signal.empty else set()
        missing_dates = all_baseline_dates - augmented_dates

        if missing_dates:
            fallback = baseline_signal[baseline_signal['date'].isin(missing_dates)]
            augmented_signal = pd.concat([augmented_signal, fallback]).sort_index()

        results[ticker] = augmented_signal
        print(f"    {dates_processed} dates augmented, {dates_fallback} fallback, {len(missing_dates)} copied from baseline")

    return results


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
    exchanges = adr_info['exchange'].unique().tolist()

    # Load futures symbols
    futures_symbols_path = data_dir / 'data' / 'raw' / 'futures_symbols.csv'
    futures_symbols = pd.read_csv(futures_symbols_path)

    # Load LASSO models
    print("\nLoading LASSO models...")
    models = load_lasso_models(model_dir)
    print(f"Loaded models for {len(models)} tickers")

    # Create close times for each exchange
    close_times = {}
    for ex in exchanges:
        close_times[ex] = (
            mcal.get_calendar(ex)
            .schedule(start_date=start_date, end_date=end_date)['market_close']
            .dt.tz_convert('America/New_York')
        ).rename('domestic_close_time')
        close_times[ex].index = close_times[ex].index.astype(str)

    # Group eligible tickers by exchange
    eligible_by_exchange = {}
    for ticker in LASSO_ELIGIBLE:
        ex = exchange_dict.get(ticker)
        if ex:
            eligible_by_exchange.setdefault(ex, []).append(ticker)

    print(f"\nLASSO-eligible tickers by exchange:")
    for ex, tickers in eligible_by_exchange.items():
        print(f"  {ex}: {tickers}")

    # Process each exchange
    all_results = {}
    for exchange_mic, tickers in eligible_by_exchange.items():
        russell_betas_path = russell_betas_dir / f'{exchange_mic}.parquet'

        results = compute_lasso_signal_for_exchange(
            exchange_mic=exchange_mic,
            eligible_tickers_on_exchange=tickers,
            models=models,
            russell_ohlcv_dir=russell_ohlcv_dir,
            russell_betas_path=russell_betas_path,
            futures_dir=futures_dir,
            adr_info=adr_info,
            futures_symbols=futures_symbols,
            close_times_df=close_times[exchange_mic],
            start_date=start_date,
            end_date=end_date,
        )
        all_results.update(results)

    # Save augmented signals for LASSO-eligible tickers
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
            print(f"  Warning: No baseline signal for {ticker}")
            continue

        dst_path.mkdir(parents=True, exist_ok=True)
        src_file = src_path / 'data.parquet'
        dst_file = dst_path / 'data.parquet'
        if src_file.exists():
            shutil.copy2(src_file, dst_file)

    # Also copy LASSO-eligible tickers that had no models
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
