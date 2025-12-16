import pandas as pd
from linux_xbbg import blp
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    adr_info_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'adr_info.csv')
    output_filename = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'earnings.csv')
    adr_info = pd.read_csv(adr_info_filename)

    result = blp.bds(adr_info['id'].tolist(), flds=['EARN_ANN_DT_TIME_HIST_WITH_EPS'])
    result.to_csv(output_filename)