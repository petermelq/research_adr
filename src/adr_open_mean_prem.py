import pandas as pd
from . import utils

if __name__ == "__main__":
    adr_df = pd.read_csv(
        "data/raw/adrs/adr_PX_OPEN_adjust_all.csv",
        index_col=0,
        parse_dates=True,
    )
    under_df = pd.read_csv(
        "data/processed/ordinary/ord_close_to_usd_adr_PX_LAST_adjust_all.csv",
        index_col=0,
        parse_dates=True,
    )

    adr_info_filename = "data/raw/adr_info.csv"
    adr_info = pd.read_csv(adr_info_filename)
    adr_dict = dict(
        zip(adr_info["id"], adr_info["adr"].str.replace(' US Equity', ''))
    )
    under_df = under_df.rename(columns=adr_dict)
    params = utils.load_params()

    premium_lookback_days = params['avg_premium_lookback_days']

    prem_df = (adr_df - under_df) / under_df
    mean_prem_df = prem_df.rolling(window=premium_lookback_days, min_periods=1).mean()
    mean_prem_df.to_csv("data/processed/adrs/adr_open_mean_premium.csv")