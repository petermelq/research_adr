import os
import pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    usdjpy_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'USDJPY_full_1min.txt')
    jpyusd_output_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'JPYUSD_full_1min.txt')
    bar_df = pd.read_csv(usdjpy_filename,
                header=None,
                names=['date','time','open','high','low','close','volume'],
    )
    bar_df[['open','close']] = 1 / bar_df[['open','close']]
    bar_df[['high','low']] = 1 / bar_df[['low','high']]
    bar_df.to_csv(jpyusd_output_filename, index=False, header=False)