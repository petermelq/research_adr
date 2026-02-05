import re
import os
import yaml
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
from glob import glob
from linux_xbbg import blp
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_partition_dates(partition_path):
    s = ','.join(glob(os.path.join(partition_path,'*=*','date=*')))
    pattern = re.compile(r"date=(?P<date>\d{4}-\d{2}-\d{2})")
    dates = pattern.findall(s)
    
    return sorted(list(set(dates)))

def get_market_business_days(calendar='NYSE'):
    '''
    Returns a Pandas CustomBusinessDay offset object that 
    excludes US stock market holidays, which can be used to 
    create date ranges that only include days when the markets
    are open.
    '''
    # Define the US stock market calendar
    cal = mcal.get_calendar(calendar)
    # Get all holidays
    holidays = cal.holidays().holidays
    if calendar in ('XNYS','NYSE'):
        holidays += (np.datetime64('2025-01-09'),) # adding Jimmy Carter holiday

    # Create a custom business day excluding US stock market holidays
    custom_bd = pd.offsets.CustomBusinessDay(holidays=holidays)

    return custom_bd

def load_params():
    '''
    Load parameters from a YAML file into a dictionary.
    '''
    param_path = os.path.join(MODULE_DIR, '..', 'params.yaml')
    with open(param_path, 'r') as f:
        params = yaml.safe_load(f)

    return params

def get_etf_holdings(etf_name, asof, ca_adj='none'):
    ##
    ## Get ETF Holdings
    ##
    bquery = blp.BlpQuery(timeout=10000).start()        
    bqlx = (
        f"for(holdings('{etf_name}', dates={asof})) "
        "get(NAME().VALUE, ID().SOURCE, ID().POSITION, ID_SEDOL1().VALUE) "
        f"with(ca_adj={ca_adj})"
    )
    df = bquery.bql(expression=bqlx)
    df = df[['security', 'field', 'value']]
    df = df.loc[(df['security'].str.contains(' Curncy')==False),:]

    ## check return format of dataframe to determine how to process:
    if 'field' in df.columns :
        df = df.pivot(index='security', columns='field', values='value').reset_index()

    #df.columns.name = None
    df = df.rename(columns={
        'id' : 'bbg_ticker',
        'ID().POSITION': 'shares',
        'NAME().VALUE': 'name',
        'TICKER().VALUE': 'ticker',
        'ID_SEDOL1().VALUE' : 'sedol',
        'ID().SOURCE': 'source',
        'security': 'id',
    })

    df['as_of_date'] = asof
    df['shares'] = pd.to_numeric(df['shares'])

    bquery = blp.BlpQuery().stop()

    return df