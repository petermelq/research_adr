import os
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression, Ridge
from linux_xbbg import blp
import sys
sys.path.append('../src/')
import utils
from utils_lasso_residuals import (
    INDEX_TO_FX_CURRENCY,
    load_fx_minute,
    compute_exchange_close_times,
    compute_fx_daily_at_close,
    convert_returns_to_usd,
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    #setting filenames
    output_filename = os.path.join(SCRIPT_DIR, '../data/processed/models/ordinary_betas_index_only.csv')
    index_filename = os.path.join(SCRIPT_DIR, '../data/processed/aligned_index_prices.csv')
    ord_filename = os.path.join(SCRIPT_DIR, '../data/raw/ordinary/ord_PX_LAST_adjust_split.csv')

    # Load ADR info
    adr_info = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/adr_info.csv'))
    adr_info = adr_info.dropna(subset=['adr'])
    adr_info = adr_info[~adr_info['id'].str.contains(' US Equity')]
    adr_info['currency'] = adr_info['currency'].replace({'GBp': 'GBP'})
    adr_info['index_future_bbg'] = adr_info['index_future_bbg'].str.strip()
    index_future_to_symbol = {'Z': 'UKX', 'VG': 'SX5E', 'NH': 'NKY'}
    adr_info['index_symbol'] = adr_info['index_future_bbg'].map(index_future_to_symbol)
    adr_info['index_currency'] = adr_info['index_symbol'].map(INDEX_TO_FX_CURRENCY)
    tickers = adr_info['id'].tolist()
    adr_dict = adr_info[['id','adr']].set_index('id')['adr'].str.replace(' US Equity', '').to_dict()

    params = utils.load_params()
    lookback_days = params['pred']['lookback_days']
    start_date = params['start_date']
    end_date = params['end_date']

    index_data = pd.read_csv(index_filename, index_col=0, parse_dates=True)

    # etf_filename = os.path.join(SCRIPT_DIR, '../data/processed/etfs/etf_mid_at_underlying_auction_adjust_all.csv')
    # etf_data = pd.read_csv(etf_filename, idx_symbol=0, parse_dates=True)
    ord_data = pd.read_csv(ord_filename, index_col=0, parse_dates=True)
    ord_data = ord_data[tickers]

    offsets_df = pd.read_csv(os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'close_time_offsets.csv'))
    exchange_offsets = dict(zip(offsets_df['exchange_mic'], offsets_df['offset']))
    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    fx_minute_cache = {}
    close_times_cache = {}
    fx_daily_cache = {}

    def get_fx_daily(exchange_mic, currency):
        key = (exchange_mic, currency)
        if key in fx_daily_cache:
            return fx_daily_cache[key]
        if currency not in fx_minute_cache:
            fx_minute_cache[currency] = load_fx_minute(currency)
        if exchange_mic not in close_times_cache:
            offset_str = exchange_offsets.get(exchange_mic, '0min')
            close_times_cache[exchange_mic] = compute_exchange_close_times(exchange_mic, offset_str, start_date, end_date)
        fx_daily_cache[key] = compute_fx_daily_at_close(fx_minute_cache[currency], close_times_cache[exchange_mic])
        return fx_daily_cache[key]

    # Calculate returns for aligned index prices (one column per ordinary ticker)
    index_returns = index_data.pct_change()
    # Rename to avoid column collision with underlying_returns
    index_returns.columns = [f'{c}_index' for c in index_returns.columns]
    underlying_returns = ord_data.pct_change()
    #etf_returns = etf_data.pct_change()

    # Align the data
    aligned_data = pd.concat([index_returns,
                              underlying_returns,
                              #etf_returns,
                              ],
                              axis=1,
                              join='inner'
                            )

    # Parameters for exponential weighting
    # hedge_df = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/hedges.csv'))
    # hedge_dict = {row['adr']: row['hedge'] for _, row in hedge_df.iterrows()}

    tickers = [adr_dict[c] for c in underlying_returns.columns]
    # Calculate exponentially-weighted rolling betas
    # betas = xr.DataArray(
    #                         coords=[
    #                             aligned_data.index,
    #                             tickers,
    #                             ['market_beta','sector_beta']
    #                         ],
    #                         dims=["date", "ticker","var_name"],
    #                         #attrs=hedge_dict,
    #                     )
    # create an empty dataframe with date and ticker as multiindex,
    betas = pd.DataFrame(
        columns=["market_beta"],
    )

    # explicitly define an empty MultiIndex with named levels
    empty_index = pd.MultiIndex(
        levels=[[], []],
        codes=[[], []],
        names=["date", "ticker"],
    )

    betas.index = empty_index

    for col in underlying_returns.columns:
        # Get the aligned index column for this stock
        idx_col = f'{col}_index'
        if idx_col not in aligned_data.columns:
            print(f"Warning: No aligned index for {col}, skipping...")
            continue

        # Get non-null data for both series
        # etf_ticker = hedge_dict[adr_dict[col]]
        # if not isinstance(etf_ticker, str) and np.isnan(etf_ticker):
        #     valid_data = aligned_data[[idx_col, col]].dropna()
        # else:
        #     valid_data = aligned_data[[idx_col, col, etf_ticker]].dropna()
        valid_data = aligned_data[[idx_col, col]].dropna()

        if len(valid_data) < 2:
            continue

        ticker_meta = adr_info[adr_info['id'] == col].iloc[0]
        exchange_mic = ticker_meta['exchange']
        stock_currency = ticker_meta['currency']
        index_currency = ticker_meta['index_currency']
        mismatch_currency = (
            isinstance(stock_currency, str)
            and isinstance(index_currency, str)
            and stock_currency != index_currency
        )

        if mismatch_currency:
            stock_fx = get_fx_daily(exchange_mic, stock_currency)
            index_fx = get_fx_daily(exchange_mic, index_currency)
            converted = pd.DataFrame(index=valid_data.index)
            converted[col] = convert_returns_to_usd(valid_data[col], stock_fx)
            converted[idx_col] = convert_returns_to_usd(valid_data[idx_col], index_fx)
            valid_data = converted.dropna()

        start_ts = pd.to_datetime(start_date)
        start_loc = int(valid_data.index.searchsorted(start_ts, side='left'))
        start_loc = max(start_loc, lookback_days + 1)

        for i in range(start_loc, len(valid_data)):
            window_start = (valid_data.index[i] - pd.Timedelta(days=lookback_days + 1)).strftime('%Y-%m-%d')
            window_end = (valid_data.index[i] - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            window_data = valid_data.loc[window_start:window_end]
            model = LinearRegression()
            model.fit(window_data[[idx_col]], window_data[col])
            market_beta = model.coef_[0]

            adr_ticker = adr_dict[col]
            betas.loc[(valid_data.index[i], adr_ticker), 'market_beta'] = market_beta
        print(f"Processed {adr_dict[col]} ({len(valid_data) - start_loc} beta rows)")

    betas = betas.reset_index().pivot(index='date', columns='ticker', values='market_beta')
    betas = betas.sort_index().sort_index(axis=1)

    # # Add index mapping information to attributes
    # betas.attrs['stock_to_index'] = stock_to_index

    # Save the betas
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    betas.to_csv(output_filename)
    print(f"Betas saved to {output_filename}")
