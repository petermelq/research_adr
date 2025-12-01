import os
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression, Ridge
from linux_xbbg import blp
import sys
sys.path.append('../src/')
import utils

if __name__ == "__main__":
    index_name = 'UKX Index'
    hold_df = pd.read_csv('../notebooks/cached_holdings/EFA_2025-09-25.csv').query('(country == "BRITAIN") & (adr_exchange != "OOTC")')
    hold_df = hold_df.dropna(subset=['adr'])
    hold_df = hold_df[~hold_df['id'].str.contains(' US Equity')]
    tickers = hold_df['id'].tolist()
    adr_dict = hold_df[['id','adr']].set_index('id')['adr'].str.replace(' US Equity', '').to_dict()
    lookback_days = 200

    exchange = 'XLON'
    cbday = utils.get_market_business_days(exchange)

    start_date = '2020-01-02'
    start_date = (pd.to_datetime(start_date) - lookback_days * cbday).strftime('%Y-%m-%d')
    end_date = '2025-11-07'

    index_filename = '../data/raw/UKX_Index_PX_LAST.csv'
    if True:#not os.path.exists(index_filename):
        index_data = blp.bdh(['UKX Index'], 'PX_LAST', start_date=start_date, end_date=end_date).droplevel(1,1)
        index_data.to_csv(index_filename)
    else:
        index_data = pd.read_csv(index_filename, index_col=0, parse_dates=True)

    etf_filename = '../data/processed/etf_mid_at_underlying_auction_adjust_all.csv'
    etf_data = pd.read_csv(etf_filename, index_col=0, parse_dates=True)

    underlying_filename = '../data/raw/brit_underlying_PX_LAST_adjust_split.csv'
    # underlying_filename = '../data/raw/brit_underlying_PX_MID_adjust_split.csv'
    if not os.path.exists(underlying_filename):
        underlying_data = blp.bdh(tickers, 'PX_LAST',
                                start_date=start_date,
                                end_date=end_date,
                                adjust='split').droplevel(1,1)
        underlying_data.to_csv(underlying_filename)
    else:
        underlying_data = pd.read_csv(underlying_filename, index_col=0, parse_dates=True)

    underlying_data = underlying_data[tickers]

    hedge_df = pd.read_csv('../data/raw/hedges.csv')
    hedge_dict = {row['adr']: row['hedge'] for _, row in hedge_df.iterrows()}

    # Calculate returns
    index_returns = index_data.pct_change()
    underlying_returns = underlying_data.pct_change()
    etf_returns = etf_data.pct_change()
    
    # Align the data
    aligned_data = pd.concat([index_returns,
                              underlying_returns,
                              etf_returns], 
                              axis=1,
                              join='inner'
                            )
    index_col = index_returns.columns[0]
    
    # Parameters for exponential weighting
    
    hedge_df = pd.read_csv('../data/raw/hedges.csv')
    hedge_dict = {row['adr']: row['hedge'] for _, row in hedge_df.iterrows()}

    tickers = [adr_dict[c] for c in underlying_returns.columns]
    # Calculate exponentially-weighted rolling betas
    betas = xr.DataArray(
                            coords=[
                                aligned_data.index, 
                                tickers,
                                ['market_beta','sector_beta']
                            ],
                            dims=["date", "ticker","var_name"],
                            attrs=hedge_dict,
                        )
    
    for col in underlying_returns.columns:
        # Get non-null data for both series
        etf_ticker = hedge_dict[adr_dict[col]]
        if not isinstance(etf_ticker, str) and np.isnan(etf_ticker):
            valid_data = aligned_data[[index_col, col]].dropna()
        else:
            valid_data = aligned_data[[index_col, col, etf_ticker]].dropna()
        
        if len(valid_data) < 2:
            continue
            
        for i in range(lookback_days, len(valid_data)):
            window_data = valid_data.iloc[i - lookback_days:i]
            model = LinearRegression()
            model.fit(window_data[[index_col]], window_data[col])
            
            market_beta = model.coef_[0]
            
            adr_ticker = adr_dict[col]
            betas.loc[valid_data.index[i], adr_ticker, 'market_beta'] = market_beta
            
            print(f"Processed {adr_dict[col]} for date {valid_data.index[i]}")
            
    betas = betas.sel(date=slice(start_date, end_date))
    # Save the betas
    output_filename = '../data/processed/brit_underlying_betas_to_UKX.nc'
    betas.to_netcdf(output_filename)
    print(f"Betas saved to {output_filename}")
    print(f"Shape: {betas.shape}")