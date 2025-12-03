import os
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression, Ridge
from linux_xbbg import blp
import sys
sys.path.append('../src/')
import utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    #setting filenames
    output_filename = os.path.join(SCRIPT_DIR, '../data/processed/models/ordinary_betas_index_only.csv')
    futures_symbols_filename = os.path.join(SCRIPT_DIR, '../data/raw/futures_symbols.csv')
    index_filename = os.path.join(SCRIPT_DIR, '../data/raw/indices/indices_PX_LAST.csv')
    ord_filename = os.path.join(SCRIPT_DIR, '../data/raw/ordinary/ord_PX_LAST_adjust_split.csv')

    # Load ADR info
    adr_info = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/raw/adr_info.csv'))
    adr_info = adr_info.dropna(subset=['adr'])
    adr_info = adr_info[~adr_info['id'].str.contains(' US Equity')]
    tickers = adr_info['id'].tolist()
    adr_dict = adr_info[['id','adr']].set_index('id')['adr'].str.replace(' US Equity', '').to_dict()
    
    # Load futures symbols mapping (index_future -> bloomberg_index)
    futures_symbols = pd.read_csv(futures_symbols_filename)
    futures_to_index = futures_symbols.set_index('bloomberg_symbol')['index'].to_dict()
    
    # Create mapping: stock -> index_future -> bloomberg_index
    stock_to_index_future = adr_info.set_index('id')['index_future_bbg'].to_dict()
    stock_to_index = {stock: futures_to_index.get(index_future)
                      for stock, index_future in stock_to_index_future.items()}
    
    # Create mapping: stock -> exchange
    stock_to_exchange = adr_info.set_index('id')['exchange'].to_dict()
    
    # Get unique indices needed
    unique_indices = list(set(stock_to_index.values()))
    unique_indices = [idx for idx in unique_indices if idx is not None]
    print(f"Indices to download: {unique_indices}")
    
    # Get unique exchanges needed
    unique_exchanges = list(set(stock_to_exchange.values()))
    unique_exchanges = [ex for ex in unique_exchanges if ex is not None]
    print(f"Exchanges: {unique_exchanges}")
    
    params = utils.load_params()
    lookback_days = params['pred']['lookback_days']
    start_date = params['start_date']
    end_date = params['end_date']
    
    index_data = pd.read_csv(index_filename, index_col=0, parse_dates=True)

    # etf_filename = os.path.join(SCRIPT_DIR, '../data/processed/etfs/etf_mid_at_underlying_auction_adjust_all.csv')
    # etf_data = pd.read_csv(etf_filename, idx_symbol=0, parse_dates=True)
    ord_data = pd.read_csv(ord_filename, index_col=0, parse_dates=True)
    ord_data = ord_data[tickers]

    # Calculate returns for all indices
    index_returns = index_data.pct_change()
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
        # Get the appropriate index for this stock
        idx_symbol = stock_to_index.get(col)
        if idx_symbol is None:
            print(f"Warning: No index found for {col}, skipping...")
            continue
    
        # Get non-null data for both series
        # etf_ticker = hedge_dict[adr_dict[col]]
        # if not isinstance(etf_ticker, str) and np.isnan(etf_ticker):
        #     valid_data = aligned_data[[idx_symbol, col]].dropna()
        # else:
        #     valid_data = aligned_data[[idx_symbol, col, etf_ticker]].dropna()
        valid_data = aligned_data[[idx_symbol, col]].dropna()
        
        if len(valid_data) < 2:
            continue
            
        padded_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        start_loc = valid_data.index.get_loc(start_date) if start_date in valid_data.index else valid_data.index.get_loc(valid_data.index[0] + pd.Timedelta(days=lookback_days+1))
        
        for i in range(start_loc, len(valid_data)):
            window_start = (valid_data.index[i] - pd.Timedelta(days=lookback_days + 1)).strftime('%Y-%m-%d')
            window_end = (valid_data.index[i] - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            window_data = valid_data.loc[window_start:window_end]
            model = LinearRegression()
            model.fit(window_data[[idx_symbol]], window_data[col])
            market_beta = model.coef_[0]
            
            adr_ticker = adr_dict[col]
            betas.loc[(valid_data.index[i], adr_ticker), 'market_beta'] = market_beta
            print(f"Processed {adr_dict[col]} against {idx_symbol} for date {valid_data.index[i]}")

    betas = betas.reset_index().pivot(index='date', columns='ticker', values='market_beta')
    betas = betas.sort_index().sort_index(axis=1)    
    
    # # Add index mapping information to attributes
    # betas.attrs['stock_to_index'] = stock_to_index
    
    # Save the betas
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    betas.to_csv(output_filename)
    print(f"Betas saved to {output_filename}")