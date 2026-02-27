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
    hedge_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'etfs', 'market', 'market_etf_PX_LAST_adjust_all.csv')
    adr_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adrs', 'adr_PX_LAST_adjust_all.csv')
    output_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'market_etf_hedge_ratios.csv')
    adr_info_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    adr_info = pd.read_csv(adr_info_filename)

    cbday = utils.get_market_business_days('NYSE')
    params = utils.load_params()
    lookback_days = params['pred']['lookback_days']
    start_date = pd.Timestamp(params['start_date']) - pd.Timedelta(days=lookback_days)
    end_date = pd.Timestamp(params['end_date'])

    adr_info['adr_tickers'] = adr_info['adr'].str.replace(' US Equity','')
    adr_tickers = adr_info['adr_tickers'].tolist()
    hedge_dict = adr_info.set_index('adr_tickers')['market_etf_hedge'].to_dict()
    
    cbday = utils.get_market_business_days('NYSE')
    dates = pd.date_range(start=start_date, end=end_date, freq=cbday)

    hedge_data = pd.read_csv(hedge_filename, index_col=0)
    adr_data = pd.read_csv(adr_filename, index_col=0)
    
    hedge_returns = hedge_data.pct_change()
    adr_returns = adr_data.pct_change()
    beta_dict = {}
    missing_pairs = []
    for ticker in adr_tickers:
        hedge_ticker = hedge_dict[ticker]
        if ticker not in adr_returns.columns or hedge_ticker not in hedge_returns.columns:
            missing_pairs.append((ticker, hedge_ticker))
            continue

        valid_data = pd.concat([adr_returns[ticker], hedge_returns[hedge_ticker]], axis=1).dropna()
        if len(valid_data) < 2:
            continue
            
        for i in range(lookback_days, len(valid_data)):
            window_data = valid_data.iloc[i - lookback_days:i]
            model = LinearRegression()
            
            hedge_model = LinearRegression()
            hedge_model.fit(window_data[[hedge_ticker]], window_data[ticker])
            hedge_beta = hedge_model.coef_[0]
            beta_dict[(valid_data.index[i], ticker)] = hedge_beta

        print(f"Processed {ticker}")

    if missing_pairs:
        preview = ", ".join([f"{t}->{h}" for t, h in missing_pairs[:10]])
        print(
            f"Warning: skipped {len(missing_pairs)} tickers due to missing ADR/hedge columns. "
            f"First pairs: {preview}"
        )

    beta_df = pd.Series(beta_dict).unstack().sort_index()
    beta_df.to_csv(output_filename)
