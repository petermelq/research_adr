import os
import pandas as pd
from linux_xbbg import blp
from . import utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MONTH_CODES = 'HMUZ'

if __name__ == '__main__':
    output_path = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'daily_futures_volume.parquet')
    params = utils.load_params()
    start_date = params['start_date']
    end_date = params['end_date']

    adr_info = pd.read_csv(os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv'))
    codes = adr_info['index_future_bbg'].unique().tolist()

    all_volume_dfs = {}
    start_year = int(pd.Timestamp(start_date).strftime('%y'))
    end_year = int(pd.Timestamp(end_date).strftime('%y')) + 1
    for code in codes:
        volume_df = blp.bdh([code + month + str(year) + ' Index'
                for month in MONTH_CODES 
                for year in range(start_year, end_year + 1)],
                'PX_VOLUME',
                start_date,
                end_date,
        ).droplevel(1,1)
        all_volume_dfs[code] = volume_df

    combined_volume_df = pd.concat(all_volume_dfs, axis=1)
    combined_volume_df.to_parquet(output_path)