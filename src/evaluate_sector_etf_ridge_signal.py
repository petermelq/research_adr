"""Evaluate sector-ETF Ridge signal vs futures-only baseline IC on sector-mapped tickers."""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('data')
ENTRY_TIMES = ['13:00', '13:30', '14:00', '14:30', '15:00', '15:30']


def load_nbbo_mid(path, col_name):
    df = pd.read_parquet(path, columns=['nbbo_bid', 'nbbo_ask'])
    out = pd.DataFrame(index=df.index)
    out[col_name] = (df['nbbo_bid'] + df['nbbo_ask']) / 2
    out['date'] = out.index.tz_localize(None).normalize()
    return out


def extract_prices_at_times(mid_df, times, value_col):
    all_dates = sorted(mid_df['date'].unique())
    res = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
    for t in times:
        s = mid_df.between_time(t, t)[value_col]
        if len(s) == 0:
            res[t] = np.nan
            continue
        idx = s.index.tz_localize(None).normalize()
        ser = pd.Series(s.values, index=idx).groupby(level=0).first()
        res[t] = ser.reindex(res.index)
    return res


def load_signal_at_times(signal_dir, tickers, entry_times):
    signals = {}
    for ticker in tickers:
        p = signal_dir / f'ticker={ticker}' / 'data.parquet'
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for et in entry_times:
            h, m = map(int, et.split(':'))
            mask = (df.index.hour == h) & (df.index.minute == m)
            sig = df.loc[mask, 'signal']
            if len(sig) == 0:
                continue
            idx = sig.index.tz_localize(None).normalize()
            ser = pd.Series(sig.values, index=idx).groupby(level=0).first()
            signals[(ticker, et)] = ser
    return signals


def compute_ic(signals, hedged_returns, tickers):
    out = {}
    for et in ENTRY_TIMES:
        s_all, r_all = [], []
        for t in tickers:
            k = (t, et)
            if k not in signals or k not in hedged_returns:
                continue
            common = signals[k].index.intersection(hedged_returns[k].index)
            if len(common) < 30:
                continue
            s = signals[k].loc[common].values
            r = hedged_returns[k].loc[common].values
            v = ~(np.isnan(s) | np.isnan(r) | np.isinf(s) | np.isinf(r))
            s = s[v]
            r = r[v]
            if len(s) < 30:
                continue
            s_all.extend(s.tolist())
            r_all.extend(r.tolist())
        if len(s_all) > 30:
            out[et] = float(np.corrcoef(np.array(s_all), np.array(r_all))[0, 1])
        else:
            out[et] = np.nan
    return pd.Series(out)


def main():
    adr_info = pd.read_csv(DATA_DIR / 'raw' / 'adr_info.csv')
    adr_info['ticker'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    ticker_to_etf = dict(zip(adr_info['ticker'], adr_info['market_etf_hedge']))

    sector_map = pd.read_csv(DATA_DIR / 'raw' / 'sector_etfs.csv')
    sector_map['adr'] = sector_map['adr'].astype(str).str.strip()
    sector_map['hedge'] = sector_map['hedge'].astype(str).str.strip()
    sector_map = sector_map.replace({'hedge': {'': pd.NA, 'nan': pd.NA}}).dropna(subset=['hedge'])
    tickers = sorted(set(sector_map['adr']).intersection(set(adr_info['ticker'])))

    hedge_ratios = pd.read_csv(DATA_DIR / 'processed' / 'market_etf_hedge_ratios.csv', index_col=0, parse_dates=True)
    close_df = pd.read_csv(DATA_DIR / 'raw' / 'adrs' / 'adr_PX_LAST_adjust_none.csv', index_col=0, parse_dates=True)
    etf_close_df = pd.read_csv(DATA_DIR / 'raw' / 'etfs' / 'market' / 'market_etf_PX_LAST_adjust_none.csv', index_col=0, parse_dates=True)

    adr_prices = {}
    for t in tickers:
        p = DATA_DIR / 'raw' / 'adrs' / 'bbo-1m' / 'nbbo' / f'ticker={t}' / 'data.parquet'
        if not p.exists():
            continue
        adr_prices[t] = extract_prices_at_times(load_nbbo_mid(p, 'mid'), ENTRY_TIMES, 'mid')

    etf_prices = {}
    for etf in sorted(set(ticker_to_etf.get(t) for t in tickers if ticker_to_etf.get(t))):
        p = DATA_DIR / 'raw' / 'etfs' / 'market' / 'bbo-1m' / 'nbbo' / f'ticker={etf}' / 'data.parquet'
        if not p.exists():
            continue
        etf_prices[etf] = extract_prices_at_times(load_nbbo_mid(p, 'etf_mid'), ENTRY_TIMES, 'etf_mid')

    hedged_returns = {}
    for t in tickers:
        if t not in adr_prices or t not in hedge_ratios.columns or t not in close_df.columns:
            continue
        etf = ticker_to_etf.get(t)
        if etf not in etf_prices or etf not in etf_close_df.columns:
            continue

        adr_px = adr_prices[t]
        etf_px = etf_prices[etf]
        adr_close = close_df[t].dropna()
        etf_close = etf_close_df[etf].dropna()
        hr = hedge_ratios[t].dropna()
        common_dates = adr_px.index.intersection(etf_px.index).intersection(adr_close.index).intersection(etf_close.index)
        if len(common_dates) == 0:
            continue

        for et in ENTRY_TIMES:
            ae = adr_px.loc[common_dates, et]
            ee = etf_px.loc[common_dates, et]
            valid = ae.notna() & ee.notna()
            d = common_dates[valid.values]
            if len(d) == 0:
                continue
            adr_ret = (adr_close.reindex(d).astype(float) - ae.loc[d].astype(float)) / ae.loc[d].astype(float)
            etf_ret = (etf_close.reindex(d).astype(float) - ee.loc[d].astype(float)) / ee.loc[d].astype(float)
            h = adr_ret - hr.reindex(d).astype(float) * etf_ret
            h = h.replace([np.inf, -np.inf], np.nan).dropna()
            if len(h):
                hedged_returns[(t, et)] = h

    ridge_signals = load_signal_at_times(DATA_DIR / 'processed' / 'index_sector_etf_ridge_signal', tickers, ENTRY_TIMES)
    base_signals = load_signal_at_times(DATA_DIR / 'processed' / 'futures_only_signal', tickers, ENTRY_TIMES)

    ridge_ic = compute_ic(ridge_signals, hedged_returns, tickers)
    base_ic = compute_ic(base_signals, hedged_returns, tickers)

    out = pd.DataFrame({'Ridge': ridge_ic, 'Baseline': base_ic})
    out['Improvement'] = out['Ridge'] - out['Baseline']

    out_dir = DATA_DIR / 'processed' / 'models' / 'with_us_stocks' / 'ridge_sector_etf_evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'pooled_ic.csv'
    out.to_csv(out_file)

    print('Pooled IC comparison (sector tickers):')
    print(out.round(4))
    print(f"\nSaved: {out_file}")
    print(f"Any positive improvement: {(out['Improvement'] > 0).any()}")


if __name__ == '__main__':
    main()
