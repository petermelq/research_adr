"""One-to-one parity evaluation with sector_ridge_signal_ic_comparison notebook logic."""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('data')
ENTRY_TIMES = ['13:00', '13:30', '14:00', '14:30', '15:00', '15:30']
OUT_DIR = DATA_DIR / 'processed' / 'models' / 'with_us_stocks' / 'ridge_sector_etf_parity_script'


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
        path = signal_dir / f'ticker={ticker}' / 'data.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for entry_time in entry_times:
            h, m = map(int, entry_time.split(':'))
            mask = (df.index.hour == h) & (df.index.minute == m)
            sig_at_time = df.loc[mask, 'signal']
            if len(sig_at_time) == 0:
                continue
            idx = sig_at_time.index.tz_localize(None).normalize()
            ser = pd.Series(sig_at_time.values, index=idx).groupby(level=0).first()
            signals[(ticker, entry_time)] = ser
    return signals


def compute_ic_filtered(signals, hedged_returns, entry_times, tickers, date_filter=None, min_obs=30):
    per_ticker_ic = pd.DataFrame(index=tickers, columns=entry_times, dtype=float)
    per_ticker_n = pd.DataFrame(index=tickers, columns=entry_times, dtype=float)
    pooled_data = {t: {'signal': [], 'actual': []} for t in entry_times}
    cs_ic_by_date = {t: {} for t in entry_times}

    for ticker in tickers:
        for entry_time in entry_times:
            key = (ticker, entry_time)
            if key not in signals or key not in hedged_returns:
                continue
            sig = signals[key]
            ret = hedged_returns[key]
            common = sig.index.intersection(ret.index)

            if date_filter is not None and key in date_filter:
                common = common.intersection(pd.Index(list(date_filter[key])))

            if len(common) < min_obs:
                continue

            s = sig.loc[common].values
            r = ret.loc[common].values
            valid = ~(np.isnan(s) | np.isnan(r) | np.isinf(s) | np.isinf(r))
            s, r = s[valid], r[valid]
            common_valid = np.array(common)[valid]

            if len(s) < min_obs:
                continue

            corr = np.corrcoef(s, r)[0, 1]
            per_ticker_ic.loc[ticker, entry_time] = corr
            per_ticker_n.loc[ticker, entry_time] = len(s)

            pooled_data[entry_time]['signal'].extend(s.tolist())
            pooled_data[entry_time]['actual'].extend(r.tolist())

            for date, sv, rv in zip(common_valid, s, r):
                if date not in cs_ic_by_date[entry_time]:
                    cs_ic_by_date[entry_time][date] = []
                cs_ic_by_date[entry_time][date].append((sv, rv))

    pooled_ic = pd.Series(index=entry_times, dtype=float, name='pooled_ic')
    pooled_n = pd.Series(index=entry_times, dtype=int, name='pooled_n')
    for t in entry_times:
        s = np.array(pooled_data[t]['signal'])
        r = np.array(pooled_data[t]['actual'])
        pooled_n[t] = len(s)
        if len(s) > min_obs:
            pooled_ic[t] = np.corrcoef(s, r)[0, 1]

    cs_ic = pd.Series(index=entry_times, dtype=float, name='cross_sectional_ic')
    cs_ic_std = pd.Series(index=entry_times, dtype=float, name='cs_ic_std')
    cs_n_days = pd.Series(index=entry_times, dtype=int, name='n_days')
    min_tickers_per_date = 5

    for t in entry_times:
        daily_ics = []
        for _, pairs in cs_ic_by_date[t].items():
            if len(pairs) < min_tickers_per_date:
                continue
            sv = np.array([x[0] for x in pairs])
            rv = np.array([x[1] for x in pairs])
            if np.std(sv) == 0 or np.std(rv) == 0:
                continue
            daily_ics.append(np.corrcoef(sv, rv)[0, 1])

        if len(daily_ics) > 0:
            cs_ic[t] = float(np.mean(daily_ics))
            cs_ic_std[t] = float(np.std(daily_ics, ddof=1) / np.sqrt(len(daily_ics))) if len(daily_ics) > 1 else np.nan
            cs_n_days[t] = len(daily_ics)

    return per_ticker_ic, per_ticker_n, pooled_ic, pooled_n, cs_ic, cs_ic_std, cs_n_days


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adr_info = pd.read_csv(DATA_DIR / 'raw' / 'adr_info.csv')
    adr_info['ticker'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    ticker_to_etf = dict(zip(adr_info['ticker'], adr_info['market_etf_hedge']))
    all_tickers = sorted(adr_info['ticker'].tolist())

    sector_map = pd.read_csv(DATA_DIR / 'raw' / 'sector_etfs.csv')
    sector_map['adr'] = sector_map['adr'].astype(str).str.strip()
    sector_map['hedge'] = sector_map['hedge'].astype(str).str.strip()
    sector_map = sector_map.replace({'hedge': {'': pd.NA, 'nan': pd.NA}}).dropna(subset=['hedge'])
    RIDGE_ELIGIBLE = sorted(set(sector_map['adr']).intersection(set(all_tickers)))

    hedge_ratios = pd.read_csv(DATA_DIR / 'processed' / 'market_etf_hedge_ratios.csv', index_col=0, parse_dates=True)
    close_df = pd.read_csv(DATA_DIR / 'raw' / 'adrs' / 'adr_PX_LAST_adjust_none.csv', index_col=0, parse_dates=True)
    etf_close_df = pd.read_csv(DATA_DIR / 'raw' / 'etfs' / 'market' / 'market_etf_PX_LAST_adjust_none.csv', index_col=0, parse_dates=True)

    adr_prices = {}
    for ticker in all_tickers:
        path = DATA_DIR / 'raw' / 'adrs' / 'bbo-1m' / 'nbbo' / f'ticker={ticker}' / 'data.parquet'
        if path.exists():
            adr_prices[ticker] = extract_prices_at_times(load_nbbo_mid(path, 'mid'), ENTRY_TIMES, 'mid')

    etf_prices = {}
    for etf in set(ticker_to_etf.values()):
        path = DATA_DIR / 'raw' / 'etfs' / 'market' / 'bbo-1m' / 'nbbo' / f'ticker={etf}' / 'data.parquet'
        if path.exists():
            etf_prices[etf] = extract_prices_at_times(load_nbbo_mid(path, 'etf_mid'), ENTRY_TIMES, 'etf_mid')

    hedged_returns = {}
    for ticker in all_tickers:
        if ticker not in adr_prices:
            continue
        etf = ticker_to_etf.get(ticker)
        if etf not in etf_prices:
            continue
        if ticker not in hedge_ratios.columns:
            continue
        if ticker not in close_df.columns:
            continue
        if etf not in etf_close_df.columns:
            continue

        adr_px = adr_prices[ticker]
        etf_px = etf_prices[etf]
        adr_close = close_df[ticker].dropna()
        etf_close = etf_close_df[etf].dropna()
        hr = hedge_ratios[ticker].dropna()

        common_dates = (
            adr_px.index.intersection(etf_px.index)
            .intersection(adr_close.index)
            .intersection(etf_close.index)
        )
        if len(common_dates) == 0:
            continue

        for entry_time in ENTRY_TIMES:
            adr_entry = adr_px.loc[common_dates, entry_time]
            etf_entry = etf_px.loc[common_dates, entry_time]
            valid = adr_entry.notna() & etf_entry.notna()
            d = common_dates[valid.values]
            if len(d) == 0:
                continue

            ae = adr_entry.loc[d].astype(float)
            ee = etf_entry.loc[d].astype(float)
            ac = adr_close.reindex(d).astype(float)
            ec = etf_close.reindex(d).astype(float)
            hr_aligned = hr.reindex(d).astype(float)

            adr_ret = (ac - ae) / ae
            etf_ret = (ec - ee) / ee
            hedged = adr_ret - hr_aligned * etf_ret
            hedged = hedged.replace([np.inf, -np.inf], np.nan).dropna()
            if len(hedged) == 0:
                continue

            hedged_returns[(ticker, entry_time)] = pd.Series(hedged.values, index=hedged.index, name=ticker)

    ridge_signals = load_signal_at_times(DATA_DIR / 'processed' / 'index_sector_etf_ridge_signal', all_tickers, ENTRY_TIMES)
    baseline_signals = load_signal_at_times(DATA_DIR / 'processed' / 'futures_only_signal', all_tickers, ENTRY_TIMES)

    ridge_active_dates = {}
    for ticker in RIDGE_ELIGIBLE:
        for entry_time in ENTRY_TIMES:
            key = (ticker, entry_time)
            if key not in ridge_signals or key not in baseline_signals:
                continue
            l = ridge_signals[key]
            b = baseline_signals[key]
            common = l.index.intersection(b.index)
            differs = l.loc[common].values != b.loc[common].values
            ridge_active_dates[key] = set(common[differs])

    ridge_per_ticker, ridge_per_n, ridge_pooled, ridge_pooled_n, ridge_cs, ridge_cs_std, ridge_cs_ndays = compute_ic_filtered(
        ridge_signals, hedged_returns, ENTRY_TIMES, RIDGE_ELIGIBLE, date_filter=ridge_active_dates
    )
    base_per_ticker, base_per_n, base_pooled, base_pooled_n, base_cs, base_cs_std, base_cs_ndays = compute_ic_filtered(
        baseline_signals, hedged_returns, ENTRY_TIMES, RIDGE_ELIGIBLE, date_filter=ridge_active_dates
    )

    ic_comparison = pd.DataFrame({
        'Ridge': ridge_pooled,
        'Baseline': base_pooled,
        'Improvement': ridge_pooled - base_pooled,
        'N obs': ridge_pooled_n,
    })

    cs_comparison = pd.DataFrame({
        'Ridge IC': ridge_cs,
        'Ridge SE': ridge_cs_std,
        'Baseline IC': base_cs,
        'Baseline SE': base_cs_std,
        'Improvement': ridge_cs - base_cs,
        'N days': ridge_cs_ndays,
    })

    main_time = '14:30'
    per_ticker_comparison = pd.DataFrame({
        'Ridge IC': ridge_per_ticker[main_time],
        'Baseline IC': base_per_ticker[main_time],
        'Improvement': ridge_per_ticker[main_time].astype(float) - base_per_ticker[main_time].astype(float),
        'N obs': ridge_per_n[main_time],
    }).dropna().sort_values('Improvement', ascending=False)

    ridge_mean_ic = ridge_per_ticker.astype(float).mean()
    base_mean_ic = base_per_ticker.astype(float).mean()
    mean_comparison = pd.DataFrame({
        'Ridge': ridge_mean_ic,
        'Baseline': base_mean_ic,
        'Improvement': ridge_mean_ic - base_mean_ic,
    })

    ic_comparison.to_csv(OUT_DIR / 'pooled_ic.csv')
    cs_comparison.to_csv(OUT_DIR / 'cross_sectional_ic.csv')
    per_ticker_comparison.to_csv(OUT_DIR / 'per_ticker_1430_ic.csv')
    mean_comparison.to_csv(OUT_DIR / 'mean_per_ticker_ic.csv')

    print('Saved parity script outputs to', OUT_DIR)


if __name__ == '__main__':
    main()
