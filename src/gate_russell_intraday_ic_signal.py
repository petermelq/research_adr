"""Gate model signal by trailing intraday IC (14:00 entry to close) per ticker/month."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-signal-dir", required=True, help="Directory with model-only signal parquet files")
    p.add_argument("--baseline-signal-dir", required=True, help="Directory with futures-only signal parquet files")
    p.add_argument("--output-dir", required=True, help="Directory for gated signal parquet files")
    p.add_argument("--adr-nbbo-dir", default="data/raw/adrs/bbo-1m/nbbo", help="ADR NBBO parquet root")
    p.add_argument(
        "--adr-close-file",
        default="data/raw/adrs/adr_PX_LAST_adjust_none.csv",
        help="ADR daily close file used as intraday exit price",
    )
    p.add_argument("--entry-time", default="14:00", help="Entry time in HH:MM (America/New_York)")
    p.add_argument("--lookback-months", type=int, default=6, help="Trailing full months for gating IC")
    p.add_argument("--min-obs", type=int, default=30, help="Minimum observations to compute IC")
    p.add_argument(
        "--min-output-date",
        default=None,
        help="Optional lower bound for output timestamps (YYYY-MM-DD)",
    )
    return p.parse_args()


def _read_signal_df(path: Path):
    if not path.exists():
        return pd.DataFrame(columns=["signal"])
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["signal"])
    if "signal" not in df.columns:
        return pd.DataFrame(columns=["signal"])
    return df


def _normalize_to_naive_dates(idx):
    dti = pd.DatetimeIndex(idx)
    if dti.tz is not None:
        dti = dti.tz_convert("America/New_York").tz_localize(None)
    return dti.normalize()


def _signal_at_time(df: pd.DataFrame, entry_time: str):
    if df.empty:
        return pd.Series(dtype=np.float32)
    hh, mm = map(int, entry_time.split(":"))
    idx = pd.DatetimeIndex(df.index)
    mask = (idx.hour == hh) & (idx.minute == mm)
    s = df.loc[mask, "signal"]
    if len(s) == 0:
        return pd.Series(dtype=np.float32)
    by_date = pd.Series(s.to_numpy(dtype=np.float32), index=_normalize_to_naive_dates(s.index))
    return by_date.groupby(level=0).first().sort_index()


def _calc_ic(sig: pd.Series, ret: pd.Series, min_obs: int):
    common = sig.index.intersection(ret.index)
    if len(common) < int(min_obs):
        return np.nan, int(len(common))
    s = sig.reindex(common).to_numpy(dtype=np.float64)
    r = ret.reindex(common).to_numpy(dtype=np.float64)
    valid = ~(np.isnan(s) | np.isnan(r) | np.isinf(s) | np.isinf(r))
    s = s[valid]
    r = r[valid]
    n = int(len(s))
    if n < int(min_obs):
        return np.nan, n
    if np.std(s) == 0 or np.std(r) == 0:
        return np.nan, n
    return float(np.corrcoef(s, r)[0, 1]), n


def _load_target_returns(ticker: str, nbbo_root: Path, close_df: pd.DataFrame, entry_time: str):
    path = nbbo_root / f"ticker={ticker}" / "data.parquet"
    if not path.exists() or ticker not in close_df.columns:
        return pd.Series(dtype=np.float32)
    px = pd.read_parquet(path, columns=["nbbo_bid", "nbbo_ask"])
    if px.empty:
        return pd.Series(dtype=np.float32)
    mid = (px["nbbo_bid"] + px["nbbo_ask"]) / 2.0
    entry = mid.between_time(entry_time, entry_time)
    if len(entry) == 0:
        return pd.Series(dtype=np.float32)
    entry_by_date = pd.Series(entry.to_numpy(dtype=np.float64), index=_normalize_to_naive_dates(entry.index))
    entry_by_date = entry_by_date.groupby(level=0).first().sort_index()

    close_s = close_df[ticker].dropna().astype(float).sort_index()
    common = entry_by_date.index.intersection(close_s.index)
    if len(common) == 0:
        return pd.Series(dtype=np.float32)
    ret = ((close_s.reindex(common) - entry_by_date.reindex(common)) / entry_by_date.reindex(common)).replace(
        [np.inf, -np.inf], np.nan
    )
    ret = ret.dropna()
    return ret.astype(np.float32).sort_index()


def _compute_month_decisions(
    model_sig_1400: pd.Series,
    base_sig_1400: pd.Series,
    realized_ret: pd.Series,
    lookback_months: int,
    min_obs: int,
):
    decisions = {}
    rows = []
    if len(model_sig_1400) == 0:
        return decisions, rows

    model_months = sorted(model_sig_1400.index.to_period("M").unique())
    for month in model_months:
        month_start = month.to_timestamp(how="start")
        hist_start = (month_start - pd.DateOffset(months=int(lookback_months))).normalize()
        hist_end = (month_start - pd.Timedelta(days=1)).normalize()

        model_hist = model_sig_1400[(model_sig_1400.index >= hist_start) & (model_sig_1400.index <= hist_end)]
        base_hist = base_sig_1400[(base_sig_1400.index >= hist_start) & (base_sig_1400.index <= hist_end)]
        ret_hist = realized_ret[(realized_ret.index >= hist_start) & (realized_ret.index <= hist_end)]

        model_ic, model_n = _calc_ic(model_hist, ret_hist, min_obs=min_obs)
        base_ic, base_n = _calc_ic(base_hist, ret_hist, min_obs=min_obs)
        use_model = bool(np.isfinite(model_ic) and np.isfinite(base_ic) and (model_ic > base_ic))
        decisions[month] = use_model
        rows.append(
            {
                "month": month.strftime("%Y-%m"),
                "model_ic": model_ic,
                "baseline_ic": base_ic,
                "improvement": (model_ic - base_ic) if (np.isfinite(model_ic) and np.isfinite(base_ic)) else np.nan,
                "model_obs": model_n,
                "baseline_obs": base_n,
                "use_model": int(use_model),
            }
        )
    return decisions, rows


def _build_gated_df(baseline_df: pd.DataFrame, model_df: pd.DataFrame, decisions: dict):
    if baseline_df.empty and model_df.empty:
        return pd.DataFrame(columns=["signal"])

    if model_df.empty:
        return baseline_df.sort_index()

    model_months = _normalize_to_naive_dates(model_df.index).to_period("M")
    use_mask = np.array([bool(decisions.get(m, False)) for m in model_months], dtype=bool)
    model_use_df = model_df.loc[use_mask]

    if baseline_df.empty:
        return model_use_df.sort_index()

    out_df = baseline_df.copy()
    if model_use_df.empty:
        return out_df.sort_index()

    common_idx = out_df.index.intersection(model_use_df.index)
    if len(common_idx) > 0:
        out_df.loc[common_idx, "signal"] = model_use_df.loc[common_idx, "signal"]
        if "date" in out_df.columns and "date" in model_use_df.columns:
            out_df.loc[common_idx, "date"] = model_use_df.loc[common_idx, "date"]

    extra_idx = model_use_df.index.difference(out_df.index)
    if len(extra_idx) > 0:
        out_df = pd.concat([out_df, model_use_df.loc[extra_idx]])
    return out_df.sort_index()


def _clip_output_date(df: pd.DataFrame, min_output_date):
    if df.empty or min_output_date is None:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        naive = idx.tz_convert("America/New_York").tz_localize(None)
    else:
        naive = idx
    return df.loc[naive >= pd.Timestamp(min_output_date)]


def main():
    args = parse_args()
    model_signal_dir = Path(args.model_signal_dir)
    baseline_signal_dir = Path(args.baseline_signal_dir)
    output_dir = Path(args.output_dir)
    nbbo_root = Path(args.adr_nbbo_dir)
    close_df = pd.read_csv(Path(args.adr_close_file), index_col=0, parse_dates=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.glob("ticker=*"):
        shutil.rmtree(p, ignore_errors=True)

    # Gate only tickers with a model signal; baseline-only tickers are intentionally excluded.
    tickers = sorted({p.name.replace("ticker=", "", 1) for p in model_signal_dir.glob("ticker=*")})

    decision_rows = []
    used_model_months = 0
    total_model_months = 0
    non_empty = 0
    for ticker in tickers:
        model_df = _read_signal_df(model_signal_dir / f"ticker={ticker}" / "data.parquet")
        baseline_df = _read_signal_df(baseline_signal_dir / f"ticker={ticker}" / "data.parquet")

        model_1400 = _signal_at_time(model_df, args.entry_time)
        base_1400 = _signal_at_time(baseline_df, args.entry_time)
        ret = _load_target_returns(ticker, nbbo_root, close_df, args.entry_time)
        decisions, rows = _compute_month_decisions(
            model_1400,
            base_1400,
            ret,
            lookback_months=args.lookback_months,
            min_obs=args.min_obs,
        )
        for r in rows:
            r["ticker"] = ticker
        decision_rows.extend(rows)
        total_model_months += len(decisions)
        used_model_months += int(sum(1 for v in decisions.values() if v))

        out_df = _build_gated_df(baseline_df, model_df, decisions)
        out_df = _clip_output_date(out_df, args.min_output_date)
        tdir = output_dir / f"ticker={ticker}"
        tdir.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(tdir / "data.parquet")
        if not out_df.empty:
            non_empty += 1

    if decision_rows:
        dec_df = pd.DataFrame(decision_rows).sort_values(["ticker", "month"])
    else:
        dec_df = pd.DataFrame(
            columns=["ticker", "month", "model_ic", "baseline_ic", "improvement", "model_obs", "baseline_obs", "use_model"]
        )
    decisions_path = output_dir / "intraday_gating_decisions.csv"
    dec_df.to_csv(decisions_path, index=False)

    print(
        f"Built intraday-gated signal for {len(tickers)} tickers; non_empty={non_empty}; "
        f"model_months_used={used_model_months}/{total_model_months}; "
        f"entry_time={args.entry_time}; lookback_months={args.lookback_months}; min_obs={args.min_obs}"
    )
    print(f"Saved decisions: {decisions_path}")


if __name__ == "__main__":
    main()
