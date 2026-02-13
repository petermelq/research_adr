"""
Utility functions for LASSO residual prediction pipeline.

This module provides functions for:
- Loading and mapping ordinary stocks to exchanges and indices
- Computing aligned returns across different market calendars
- Residualizing returns with respect to indices
- Computing existing beta model residuals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from joblib import Parallel, delayed

__script_dir__ = Path(__file__).parent.absolute()


# Mapping from index futures (bloomberg_symbol in adr_info.csv) to underlying
# cash index symbols (from futures_symbols.csv 'index' column)
INDEX_FUTURE_TO_SYMBOL = {
    'NH': 'NKY',    # Nikkei 225 (Japan)
    'Z': 'UKX',     # FTSE 100 (UK)
    'VG': 'SX5E',   # Euro Stoxx 50 (Europe)
}


def load_ordinary_exchange_mapping():
    """
    Load mapping from ordinary ticker to exchange MIC.

    Excludes XTKS (Tokyo) and XASX (Australia) as we don't have Russell data.

    Returns:
        dict: {ordinary_ticker: exchange_mic}
        dict: {ordinary_ticker: adr_ticker}
    """
    adr_info_path = __script_dir__ / '..' / 'data' / 'raw' / 'adr_info.csv'
    adr_info = pd.read_csv(adr_info_path)

    # Exclude exchanges we don't have Russell data for
    excluded_exchanges = ['XTKS', 'XASX']
    adr_info = adr_info[~adr_info['exchange'].isin(excluded_exchanges)]

    # Create mappings
    ordinary_to_exchange = dict(zip(adr_info['id'], adr_info['exchange']))

    # Extract ADR ticker (remove ' US Equity' suffix)
    adr_info['adr_ticker'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    ordinary_to_adr = dict(zip(adr_info['id'], adr_info['adr_ticker']))

    return ordinary_to_exchange, ordinary_to_adr


def load_index_mapping():
    """
    Load mapping from ordinary ticker to index symbol.

    Returns:
        dict: {ordinary_ticker: index_symbol}
        dict: {exchange_mic: index_symbol}
    """
    adr_info_path = __script_dir__ / '..' / 'data' / 'raw' / 'adr_info.csv'
    adr_info = pd.read_csv(adr_info_path)

    # Exclude exchanges we don't have Russell data for
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


def compute_aligned_returns(prices, dates=None):
    """
    Compute returns handling missing dates (holidays).

    For each target date, finds the most recent previous date with data
    and computes the return from that date to the target date.

    Args:
        prices: DataFrame with dates as index, tickers as columns
        dates: Optional list of target dates. If None, uses all dates in prices.

    Returns:
        DataFrame with returns aligned to target dates
    """
    if dates is None:
        dates = prices.index

    # Convert to pandas DatetimeIndex if not already
    dates = pd.DatetimeIndex(dates)
    prices_index = pd.DatetimeIndex(prices.index)

    # Initialize returns dataframe
    returns = pd.DataFrame(index=dates, columns=prices.columns, dtype=float)

    for date in dates:
        # Find the most recent date with data on or before this date
        available_dates = prices_index[prices_index <= date]

        if len(available_dates) >= 2:
            # Use the most recent two dates
            prev_date = available_dates[-2]
            curr_date = available_dates[-1]

            # Compute return
            prev_prices = prices.loc[prev_date]
            curr_prices = prices.loc[curr_date]

            # Handle missing values: if either is NaN, return is NaN
            ret = (curr_prices - prev_prices) / prev_prices
            returns.loc[date] = ret

    return returns


def _residualize_single_ticker(ticker, ticker_returns, index_aligned, common_dates, window=60):
    """
    Residualize a single ticker's returns (helper for parallel processing).

    Args:
        ticker: Ticker name
        ticker_returns: Series of returns for this ticker
        index_aligned: Series of index returns (aligned to common dates)
        common_dates: DatetimeIndex of common dates
        window: Number of days for rolling beta calculation

    Returns:
        Series of residualized returns for this ticker
    """
    residuals = pd.Series(index=common_dates, dtype=float, name=ticker)
    ticker_returns = ticker_returns.dropna()

    # For each date, compute beta using prior window days
    for i in range(len(common_dates)):
        date = common_dates[i]

        # Skip if not enough history
        if i < window:
            continue

        # Get historical returns for beta calculation
        start_idx = max(0, i - window)
        hist_ticker = ticker_returns.iloc[start_idx:i]
        hist_index = index_aligned.iloc[start_idx:i]

        # Align historical data (only use dates where both are available)
        hist_dates = hist_ticker.index.intersection(hist_index.index)
        if len(hist_dates) < 20:  # Require at least 20 observations
            continue

        hist_ticker_aligned = hist_ticker.loc[hist_dates]
        hist_index_aligned = hist_index.loc[hist_dates]

        # Drop NaN from both series before computing cov/var
        valid_mask = hist_ticker_aligned.notna() & hist_index_aligned.notna()
        hist_ticker_clean = hist_ticker_aligned[valid_mask]
        hist_index_clean = hist_index_aligned[valid_mask]

        if len(hist_ticker_clean) < 20:
            continue

        # Compute beta using covariance
        cov = np.cov(hist_ticker_clean, hist_index_clean)[0, 1]
        var = np.var(hist_index_clean)

        if var > 0:
            beta = cov / var
        else:
            beta = 0

        # Compute residual for current date
        if date in ticker_returns.index and date in index_aligned.index:
            curr_return = ticker_returns.loc[date]
            curr_index_return = index_aligned.loc[date]
            if not (np.isnan(curr_return) or np.isnan(curr_index_return)):
                residuals.loc[date] = curr_return - beta * curr_index_return

    return residuals


def residualize_returns(returns, index_returns, window=60, n_jobs=4):
    """
    Residualize returns with respect to index using rolling beta.

    Uses parallel processing to speed up computation across tickers.

    For each date, computes beta using the prior 'window' days,
    then residualizes: residual = return - beta * index_return

    Args:
        returns: DataFrame with dates as index, tickers as columns
        index_returns: Series with dates as index, index returns as values
        window: Number of days for rolling beta calculation
        n_jobs: Number of parallel jobs (default 28, use -1 for all cores)

    Returns:
        DataFrame with residualized returns (same shape as input)
    """
    # Align indices
    common_dates = returns.index.intersection(index_returns.index)
    returns_aligned = returns.loc[common_dates]
    index_aligned = index_returns.loc[common_dates]

    # Parallel computation across tickers
    residuals_list = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_residualize_single_ticker)(
            ticker,
            returns_aligned[ticker],
            index_aligned,
            common_dates,
            window
        )
        for ticker in returns.columns
    )

    # Combine results into DataFrame
    residuals = pd.concat(residuals_list, axis=1)

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
