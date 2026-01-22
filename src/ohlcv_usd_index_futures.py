import os
import argparse
import pandas as pd
import polars as pl
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert OHLCV futures data to USD using FX rates.')
    parser.add_argument('fut_dir', type=str, help='Directory containing futures minute bar data.')
    parser.add_argument('output_dir', type=str, help='Directory to save converted futures data.')
    args = parser.parse_args()
    fut_dir = args.fut_dir
    futures_filenames = [
        os.path.join(args.fut_dir, 'exchange=IFLL.IMPACT', 'code=Z'),
        os.path.join(args.fut_dir, 'exchange=XEUR.EOBI', 'code=FESX'),
        os.path.join(args.fut_dir, 'exchange=GLBX.MDP3', 'code=NIY'),
    ]
    fx_filenames = [
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'GBPUSD_full_1min.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'EURUSD_full_1min.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'JPYUSD_full_1min.txt'),    
    ]
    futures_symbols = pd.read_csv(os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures_symbols.csv'))
    fr_map = {row['exchange_symbol']: row['first_rate_symbol'] for _, row in futures_symbols.iterrows()}
    notional_multipliers = [
        # currency per point
        10,   # FTUK
        #25,   # FDAX
        #10,   # FCE
        #50,   # FXXP
        10,   # FESX
     #   200,  # FTI
        500,  # NIY
    ]
    for futures_filename, fx_filename, notional_multiplier in zip(futures_filenames,
                                                                    fx_filenames,
                                                                    notional_multipliers,
                                                                ):
        futures_symbol = fr_map[os.path.basename(futures_filename).split('code=')[-1]]
        output_filename = os.path.join(
                    args.output_dir,
                    f'symbol={futures_symbol}',
                    f'{futures_symbol}_close_to_usd_1min.parquet',
                )
        # Read FX data with polars for better memory efficiency
        fx_df = pl.read_csv(
            fx_filename,
            has_header=False,
            new_columns=['date','time','open','high','low','close','volume']
        )
        
        # Create timestamp column efficiently in polars (format: YYYYMMDD,HH:MM:SS)
        fx_df = fx_df.with_columns([
            pl.concat_str([
                pl.col('date').cast(pl.Utf8),
                pl.lit(' '),
                pl.col('time').cast(pl.Utf8)
            ]).str.to_datetime(format='%Y%m%d %H:%M:%S', time_unit='ns').dt.replace_time_zone('America/New_York').alias('timestamp')
        ]).select(['timestamp', pl.col('close').alias('fx_rate')])
        
        # Read futures data with polars
        futures_df = pl.read_parquet(
            futures_filename,
        )
        
        # Apply notional multiplier and parse timestamp (format: YYYY-MM-DD HH:MM:SS)
        futures_df = futures_df.with_columns([
            pl.col('ts_event').dt.replace_time_zone('America/New_York'),
            (pl.col(['open','high','low','close']) * notional_multiplier)
        ]) # Perform the join efficiently with polars
        futures_df = futures_df.join(fx_df, left_on='ts_event', right_on='timestamp', how='left')
        
        # Apply FX conversion
        futures_df = futures_df.with_columns([
            (pl.col(['open','high','low','close']) * pl.col('fx_rate'))
        ])
        
        futures_df = futures_df.rename({'symbol':'contract_symbol','ts_event':'timestamp'})

        # Write to CSV (polars handles this more efficiently than pandas)
        output_dirname = os.path.dirname(output_filename)
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname, exist_ok=True)

        futures_df.write_parquet(output_filename)