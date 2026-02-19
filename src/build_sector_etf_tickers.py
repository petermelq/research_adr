import pandas as pd
from pathlib import Path

__script_dir__ = Path(__file__).parent.absolute()


def main():
    in_path = __script_dir__ / '..' / 'data' / 'raw' / 'sector_etfs.csv'
    out_path = __script_dir__ / '..' / 'data' / 'processed' / 'sector_etfs' / 'sector_etf_tickers.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    tickers = (
        df['hedge']
        .dropna()
        .astype(str)
        .str.strip()
        .replace('', pd.NA)
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    pd.DataFrame({'ticker': tickers.values}).to_csv(out_path, index=False)
    print(f"Saved {len(tickers)} unique sector ETF tickers to {out_path}")


if __name__ == '__main__':
    main()
