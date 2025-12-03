import os
import polars as pl
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    futures_filenames = [
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'minute_bars', 'FTUK_full_1min_continuous_ratio_adjusted.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'minute_bars', 'FDAX_full_1min_continuous_ratio_adjusted.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'minute_bars', 'FCE_full_1min_continuous_ratio_adjusted.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'minute_bars', 'FXXP_full_1min_continuous_ratio_adjusted.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'minute_bars', 'FESX_full_1min_continuous_ratio_adjusted.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'minute_bars', 'FTI_full_1min_continuous_ratio_adjusted.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'futures', 'minute_bars', 'NIY_full_1min_continuous_ratio_adjusted.txt'),
    ]
    fx_filenames = [
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'GBPUSD_full_1min.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'EURUSD_full_1min.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'EURUSD_full_1min.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'EURUSD_full_1min.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'EURUSD_full_1min.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'EURUSD_full_1min.txt'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'currencies', 'minute_bars', 'JPYUSD_full_1min.txt'),
    ]
    notional_multipliers = [
        # currency per point
        10,   # FTUK
        25,   # FDAX
        10,   # FCE
        50,   # FXXP
        10,   # FESX
        200,  # FTI
        500,  # NIY
    ]
    for futures_filename, fx_filename, notional_multiplier in zip(futures_filenames,
                                                                    fx_filenames,
                                                                    notional_multipliers,
                                                                ):
        futures_symbol = os.path.basename(futures_filename).split('_full_')[0]
        output_filename = os.path.join(
                    SCRIPT_DIR,
                    '..',
                    'data',
                    'processed',
                    'futures',
                    'converted_minute_bars',
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
            ]).str.to_datetime(format='%Y%m%d %H:%M:%S').dt.replace_time_zone('America/New_York').alias('timestamp')
        ]).select(['timestamp', pl.col('close').alias('fx_rate')])
        
        # Read futures data with polars
        futures_df = pl.read_csv(
            futures_filename,
            has_header=False,
            new_columns=['timestamp','open','high','low','close','volume']
        )
        
        # Apply notional multiplier and parse timestamp (format: YYYY-MM-DD HH:MM:SS)
        futures_df = futures_df.with_columns([
            pl.col('timestamp').str.to_datetime(format='%Y-%m-%d %H:%M:%S').dt.replace_time_zone('America/New_York'),
            (pl.col(['open','high','low','close']) * notional_multiplier)
        ])    # Perform the join efficiently with polars
        futures_df = futures_df.join(fx_df, on='timestamp', how='left')
        
        # Apply FX conversion
        futures_df = futures_df.with_columns([
            (pl.col(['open','high','low','close']) * pl.col('fx_rate'))
        ])

        # Write to CSV (polars handles this more efficiently than pandas)
        output_dirname = os.path.dirname(output_filename)
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname, exist_ok=True)

        futures_df.write_parquet(output_filename)