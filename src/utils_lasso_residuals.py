"""
Utility functions for LASSO residual prediction pipeline.

This module provides functions for:
- Loading and mapping ordinary stocks to exchanges and indices
- Computing aligned returns across different market calendars
- Residualizing returns with respect to indices
- Computing existing beta model residuals
"""

import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
from pathlib import Path
import os

__script_dir__ = Path(__file__).parent.absolute()


# Mapping from index futures (bloomberg_symbol in adr_info.csv) to underlying
# cash index symbols (from futures_symbols.csv 'index' column)
INDEX_FUTURE_TO_SYMBOL = {
    'NH': 'NKY',    # Nikkei 225 (Japan)
    'Z': 'UKX',     # FTSE 100 (UK)
    'VG': 'SX5E',   # Euro Stoxx 50 (Europe)
}

# Mapping from cash index to FX currency for USD conversion.
INDEX_TO_FX_CURRENCY = {
    'SX5E': 'EUR',  # EURUSD
    'UKX': 'GBP',   # GBPUSD
    'NKY': 'JPY',   # JPYUSD
}

def load_ordinary_exchange_mapping(include_asia=False):
    """
    Load mapping from ordinary ticker to exchange MIC.

    Returns:
        dict: {ordinary_ticker: exchange_mic}
        dict: {ordinary_ticker: adr_ticker}
    """
    adr_info_path = __script_dir__ / '..' / 'data' / 'raw' / 'adr_info.csv'
    adr_info = pd.read_csv(adr_info_path)

    if not include_asia:
        excluded_exchanges = ['XTKS', 'XASX']
        adr_info = adr_info[~adr_info['exchange'].isin(excluded_exchanges)]

    # Create mappings
    ordinary_to_exchange = dict(zip(adr_info['id'], adr_info['exchange']))

    # Extract ADR ticker (remove ' US Equity' suffix)
    adr_info['adr_ticker'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    ordinary_to_adr = dict(zip(adr_info['id'], adr_info['adr_ticker']))

    return ordinary_to_exchange, ordinary_to_adr


def load_index_mapping(include_asia=False):
    """
    Load mapping from ordinary ticker to index symbol.

    Returns:
        dict: {ordinary_ticker: index_symbol}
        dict: {exchange_mic: index_symbol}
    """
    adr_info_path = __script_dir__ / '..' / 'data' / 'raw' / 'adr_info.csv'
    adr_info = pd.read_csv(adr_info_path)

    if not include_asia:
        excluded_exchanges = ['XTKS', 'XASX']
        adr_info = adr_info[~adr_info['exchange'].isin(excluded_exchanges)]

    # Strip whitespace from index_future_bbg before mapping
    adr_info['index_future_bbg'] = adr_info['index_future_bbg'].str.strip()

    # Map index future to index symbol
    adr_info['index_symbol'] = adr_info['index_future_bbg'].map(INDEX_FUTURE_TO_SYMBOL)

    # Create mappings
    ordinary_to_index = dict(zip(adr_info['id'], adr_info['index_symbol']))
    exchange_to_index = dict(zip(adr_info['exchange'], adr_info['index_symbol']))

    return ordinary_to_index, exchange_to_index


def load_fx_minute(currency):
    """
    Load FX minute bar data for a currency pair (XXXUSD).

    Args:
        currency: Currency code (e.g., 'EUR', 'GBP')

    Returns:
        Series with tz-aware ET timestamp index and close rate values
    """
    fx_dir = __script_dir__ / '..' / 'data' / 'raw' / 'currencies' / 'minute_bars'
    fx_file = fx_dir / f'{currency}USD_full_1min.txt'
    fx_df = pd.read_csv(
        fx_file, header=None, index_col=None,
        names=['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    )
    fx_df['timestamp'] = pd.to_datetime(
        fx_df['date'].astype(str) + ' ' + fx_df['time'].astype(str)
    ).dt.tz_localize('America/New_York')
    return fx_df.set_index('timestamp')['close']


def compute_exchange_close_times(exchange_mic, offset_str, start_date, end_date):
    """
    Compute tz-aware ET close times for an exchange (normal close days only).

    Args:
        exchange_mic: Exchange MIC code (e.g., 'XLON')
        offset_str: Offset string compatible with pd.Timedelta (e.g., '6min')
        start_date: Start date
        end_date: End date

    Returns:
        Series indexed by date (datetime) with tz-aware ET close times as values
    """
    cal = mcal.get_calendar(exchange_mic)
    sched = cal.schedule(start_date=start_date, end_date=end_date)
    close_times_et = sched['market_close'].dt.tz_convert('America/New_York')

    # Filter to normal close days only
    close_times_only = close_times_et.dt.time
    most_common_close = close_times_only.mode()[0]
    is_normal_close = close_times_only == most_common_close
    close_times_et = close_times_et[is_normal_close]

    # Add offset
    close_times_et = close_times_et + pd.Timedelta(offset_str)

    return close_times_et


def compute_fx_daily_at_close(fx_minute, close_times):
    """
    Compute daily FX returns using FX rate at exchange close time.

    For each date, looks up the FX rate at or before the exchange close time
    using binary search, then computes close-to-close FX returns.

    Args:
        fx_minute: Series with tz-aware ET timestamp index and close rate values
        close_times: Series indexed by date with tz-aware ET close times

    Returns:
        Series indexed by date with daily FX returns
    """
    fx_idx_int = fx_minute.index.values.astype('int64')
    ct_int = close_times.values.astype('int64')
    # searchsorted(side='right') gives index of first element > close_time
    # subtract 1 to get last element <= close_time
    indices = np.searchsorted(fx_idx_int, ct_int, side='right') - 1
    valid = indices >= 0
    fx_at_close = pd.Series(
        data=fx_minute.values[indices[valid]],
        index=close_times.index[valid],
        dtype=float,
    )
    return fx_at_close.pct_change()


def convert_returns_to_usd(native_returns, fx_returns):
    """
    Convert native-currency returns to USD.

    Formula: r_usd = (1 + r_native) * (1 + r_fx) - 1

    Args:
        native_returns: Series or DataFrame of returns in native currency
        fx_returns: Series of FX returns (XXXUSD), indexed by date

    Returns:
        Same type as native_returns, with USD-converted returns
    """
    if isinstance(native_returns, pd.DataFrame):
        common_dates = native_returns.index.intersection(fx_returns.index)
        fx_aligned = fx_returns.loc[common_dates]
        native_aligned = native_returns.loc[common_dates]
        return (1 + native_aligned).multiply(1 + fx_aligned, axis=0) - 1
    else:
        common_dates = native_returns.index.intersection(fx_returns.index)
        fx_aligned = fx_returns.loc[common_dates]
        native_aligned = native_returns.loc[common_dates]
        return (1 + native_aligned) * (1 + fx_aligned) - 1


def compute_aligned_returns(prices, dates=None):
    """
    Compute returns handling missing dates (holidays).

    For each target date, finds the most recent previous date with data
    and computes the return from that date to the target date.
    Uses vectorized binary search for efficiency.

    Args:
        prices: DataFrame with dates as index, tickers as columns
        dates: Optional list of target dates. If None, uses all dates in prices.

    Returns:
        DataFrame with returns aligned to target dates
    """
    if dates is None:
        dates = prices.index

    dates = pd.DatetimeIndex(dates)
    prices_int = prices.index.values.astype('int64')
    target_int = dates.values.astype('int64')

    # searchsorted(side='right') gives index of first element > target
    # so idx-1 is the most recent date <= target (curr), idx-2 is previous
    idx = np.searchsorted(prices_int, target_int, side='right')
    valid = idx >= 2

    returns = pd.DataFrame(np.nan, index=dates, columns=prices.columns, dtype=float)

    if valid.any():
        curr_idx = idx[valid] - 1
        prev_idx = idx[valid] - 2
        curr_prices = prices.values[curr_idx]
        prev_prices = prices.values[prev_idx]
        ret_vals = (curr_prices - prev_prices) / prev_prices
        returns.values[np.where(valid)[0]] = ret_vals

    return returns


def residualize_returns(returns, index_returns, window=60):
    """
    Residualize returns with respect to index using rolling beta.

    For each date, computes beta using the prior 'window' days via
    vectorized rolling covariance, then residualizes:
        residual = return - beta * index_return

    Args:
        returns: DataFrame with dates as index, tickers as columns
        index_returns: Series with dates as index, index returns as values
        window: Number of days for rolling beta calculation

    Returns:
        DataFrame with residualized returns (same shape as input)
    """
    common_dates = returns.index.intersection(index_returns.index)
    R = returns.loc[common_dates]
    I = index_returns.loc[common_dates]

    # Rolling population covariance via E[R*I] - E[R]*E[I]
    RI = R.multiply(I, axis=0)
    roll_mean_RI = RI.rolling(window, min_periods=20).mean()
    roll_mean_R = R.rolling(window, min_periods=20).mean()
    roll_mean_I = I.rolling(window, min_periods=20).mean()
    roll_cov = roll_mean_RI.subtract(roll_mean_R.multiply(roll_mean_I, axis=0))

    # Rolling population variance of index
    idx_var = I.rolling(window, min_periods=20).var(ddof=0)

    # Compute betas = cov / var
    betas = roll_cov.divide(idx_var, axis=0)
    betas[idx_var == 0] = 0

    # Residualize
    residuals = R - betas.multiply(I, axis=0)

    return residuals


def get_existing_beta_residuals(ordinary_ticker, adr_ticker, ordinary_returns,
                                index_returns, betas_df):
    """
    Compute residuals using existing beta model.

    Args:
        ordinary_ticker: Ticker for the ordinary stock
        adr_ticker: Corresponding ADR ticker (used to look up beta)
        ordinary_returns: Series with ordinary stock returns
        index_returns: Series with index returns
        betas_df: DataFrame with time-varying betas (dates x tickers)

    Returns:
        Series with residuals (actual - predicted returns)
    """
    # Align dates
    common_dates = ordinary_returns.index.intersection(index_returns.index).intersection(betas_df.index)

    if len(common_dates) == 0:
        return pd.Series(dtype=float)

    # Get aligned data
    ordinary_aligned = ordinary_returns.loc[common_dates]
    index_aligned = index_returns.loc[common_dates]

    # Get betas for this ticker
    if adr_ticker not in betas_df.columns:
        print(f"Warning: {adr_ticker} not found in betas, using beta=1.0")
        betas_aligned = pd.Series(1.0, index=common_dates)
    else:
        betas_aligned = betas_df.loc[common_dates, adr_ticker]

    # Compute predicted returns
    predicted_returns = betas_aligned * index_aligned

    # Compute residuals
    residuals = ordinary_aligned - predicted_returns

    return residuals


def fill_missing_values(df, fill_value=0.0):
    """
    Fill missing values in DataFrame.

    Args:
        df: DataFrame with potential NaN values
        fill_value: Value to use for filling (default 0.0)

    Returns:
        DataFrame with NaN values filled
    """
    return df.fillna(fill_value)
