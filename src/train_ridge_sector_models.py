"""Train Ridge models on sector-ETF feature set."""

from pathlib import Path
import pandas as pd
from train_ridge_models import train_rolling_models

__script_dir__ = Path(__file__).parent.absolute()


def main():
    features_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'features_sector_etf'
    models_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'ridge_sector_etf'
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(features_dir.glob('*.parquet'))
    print(f"Found {len(feature_files)} sector feature files")
    all_metadata = []

    for i, feature_file in enumerate(feature_files, start=1):
        ticker = feature_file.stem
        print(f"[{i}/{len(feature_files)}] {ticker}")
        try:
            features = pd.read_parquet(feature_file)
            metadata = train_rolling_models(ticker, features, models_dir)
            all_metadata.extend(metadata)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if all_metadata:
        md = pd.DataFrame(all_metadata)
        out = models_dir / 'training_metadata.csv'
        md.to_csv(out, index=False)
        print(f"Saved metadata: {out} ({len(md)} models, {md['ticker'].nunique()} tickers)")
    else:
        print("No sector ridge models trained")


if __name__ == '__main__':
    main()
