import yaml
import argparse
import pandas as pd
from linux_xbbg import blp
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Bloomberg daily data')
    parser.add_argument('tickers_filename', type=str, help='Path to CSV file with tickers')
    parser.add_argument('output_filename', type=str, help='Output CSV filename')
    parser.add_argument('--field', type=str, help='Field to download from Bloomberg')
    parser.add_argument('--start_date', default=None, type=str, help='Start date for data download (YYYY-MM-DD). Defaults to params.yaml value')
    parser.add_argument('--end_date', default=None, type=str, help='End date for data download (YYYY-MM-DD). Defaults to params.yaml value')
    parser.add_argument('--tickers_columns', nargs='+', type=str, default=['ticker'], help='Column name in CSV containing tickers')
    parser.add_argument('--symbol_suffix', type=str, default='', help='Suffix to append to each ticker for Bloomberg')
    parser.add_argument('--include_suffix', action='store_true', help='Whether to include the suffix in the output tickers')
    parser.add_argument('--pad_lookback', type=int, default=0, help='Number of extra days to pad before start_date for data download. If > 0, start_date is adjusted accordingly, with an extra day added to the price series, so that the return series has a pad with "pad_lookback" days.')
    parser.add_argument('--adjust', type=str, default='none', help='Adjustment type for Bloomberg data')
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, '..', 'params.yaml'), 'r') as f:
        params = yaml.safe_load(f)

    tickers_df = pd.read_csv(args.tickers_filename)
    tickers = list(set([ticker for col in args.tickers_columns for ticker in tickers_df[col].dropna().to_list()]))
    bbg_tickers = [t + args.symbol_suffix for t in tickers]
    
    start_date = args.start_date if args.start_date else params['start_date']
    if args.pad_lookback > 0:
        start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=args.pad_lookback + 1)).strftime('%Y-%m-%d')

    end_date = args.end_date if args.end_date else params['end_date']
    
    data = blp.bdh(bbg_tickers,
                    [args.field],
                    start_date=start_date,
                    end_date=end_date,
                    adjust=args.adjust,
                ).droplevel(1,1)
    if not args.include_suffix:
        data.columns = [col.split()[0] for col in data.columns]
    
    output_dirname = os.path.dirname(args.output_filename)
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname, exist_ok=True)
    
    data.to_csv(args.output_filename)