"""Compare notebook parity outputs vs parity script outputs."""

from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path('data') / 'processed' / 'models' / 'with_us_stocks' / 'ridge_sector_etf_parity_script'
NOTEBOOK_DIR = Path('data') / 'processed' / 'models' / 'with_us_stocks' / 'ridge_sector_etf_parity_notebook'
FILES = [
    'pooled_ic.csv',
    'cross_sectional_ic.csv',
    'per_ticker_1430_ic.csv',
    'mean_per_ticker_ic.csv',
]


def load_df(p: Path):
    df = pd.read_csv(p)
    # normalize index written in different ways (index column vs named column)
    if 'Entry Time' in df.columns:
        df = df.set_index('Entry Time')

    if 'ticker' in df.columns and 'N obs' in df.columns and 'Entry Time' not in df.columns:
        # per-ticker table should compare row-wise by ticker
        df = df.set_index('ticker')

    # normalize unnamed index column if present
    unnamed = [c for c in df.columns if c.startswith('Unnamed:')]
    if unnamed:
        df = df.rename(columns={unnamed[0]: 'index'}).set_index('index')
    return df.sort_index().sort_index(axis=1)


def main():
    tol = 1e-5
    all_ok = True

    for fn in FILES:
        sp = SCRIPT_DIR / fn
        npf = NOTEBOOK_DIR / fn
        if not sp.exists() or not npf.exists():
            raise FileNotFoundError(f'Missing file for comparison: {sp} or {npf}')

        s = load_df(sp)
        n = load_df(npf)

        # align shapes/indices
        if list(s.columns) != list(n.columns) or list(s.index) != list(n.index):
            print(f'[FAIL] {fn}: index/column mismatch')
            print('script cols:', list(s.columns))
            print('nb cols    :', list(n.columns))
            print('script idx head:', list(s.index)[:5])
            print('nb idx head    :', list(n.index)[:5])
            all_ok = False
            continue

        # compare numerics and strings safely
        diff_found = False
        for col in s.columns:
            if pd.api.types.is_numeric_dtype(s[col]) and pd.api.types.is_numeric_dtype(n[col]):
                sv = s[col].astype(float).values
                nv = n[col].astype(float).values
                mask = ~(np.isnan(sv) & np.isnan(nv))
                if mask.any():
                    if np.nanmax(np.abs(sv[mask] - nv[mask])) > tol:
                        diff_found = True
                        break
            else:
                if not s[col].astype(str).equals(n[col].astype(str)):
                    diff_found = True
                    break

        if diff_found:
            print(f'[FAIL] {fn}: values differ')
            all_ok = False
        else:
            print(f'[OK]   {fn}')

    if not all_ok:
        raise SystemExit(1)

    print('Parity check passed: notebook and pipeline outputs agree exactly within tolerance.')


if __name__ == '__main__':
    main()
