"""Build gated signal by selecting model or baseline signal per model test window."""

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Directory with per-ticker model pickle files")
    p.add_argument("--model-signal-dir", required=True, help="Directory with model-augmented signal parquet files")
    p.add_argument("--baseline-signal-dir", required=True, help="Directory with futures-only baseline signal parquet files")
    p.add_argument("--output-dir", required=True, help="Directory for gated signal parquet files")
    return p.parse_args()


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ticker_windows(model_dir: Path):
    windows_by_ticker = {}
    for tdir in sorted(model_dir.iterdir()):
        if not tdir.is_dir():
            continue
        ticker = tdir.name
        windows = []
        for mf in sorted(tdir.glob("*.pkl")):
            md = _load_pickle(mf)
            ts, te = map(pd.Timestamp, md["test_period"])
            model_val_ic = md.get("val_ic")
            baseline_val_ic = md.get("baseline_val_ic")
            if model_val_ic is None:
                model_val_ic = md.get("train_ic", np.nan)
            if baseline_val_ic is None:
                baseline_val_ic = md.get("baseline_train_ic", 0.0)
            model_val_ic = float(model_val_ic)
            baseline_val_ic = float(baseline_val_ic)
            windows.append(
                {
                    "start": ts.normalize(),
                    "end": te.normalize(),
                    "use_model": bool(model_val_ic > baseline_val_ic),
                }
            )
        if windows:
            windows_by_ticker[ticker] = windows
    return windows_by_ticker


def _read_signal_df(path: Path):
    if not path.exists():
        return pd.DataFrame(columns=["signal", "date"])
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["signal", "date"])
    return df


def _model_mask_for_windows(model_df: pd.DataFrame, windows):
    if model_df.empty or not windows:
        return np.zeros(0, dtype=bool)
    model_dates = pd.DatetimeIndex(model_df.index).tz_localize(None).normalize()
    use_mask = np.zeros(len(model_df), dtype=bool)
    for w in windows:
        if not w["use_model"]:
            continue
        use_mask |= (model_dates >= w["start"]) & (model_dates <= w["end"])
    return use_mask


def build_gated_ticker_df(baseline_df: pd.DataFrame, model_df: pd.DataFrame, windows):
    if baseline_df.empty and model_df.empty:
        return pd.DataFrame(columns=["signal", "date"])

    model_use_mask = _model_mask_for_windows(model_df, windows)
    model_use_df = model_df.loc[model_use_mask] if len(model_use_mask) else pd.DataFrame(columns=["signal", "date"])

    if baseline_df.empty:
        out_df = model_use_df.copy()
        return out_df.sort_index()

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


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_signal_dir = Path(args.model_signal_dir)
    baseline_signal_dir = Path(args.baseline_signal_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.glob("ticker=*"):
        shutil.rmtree(p, ignore_errors=True)

    windows_by_ticker = load_ticker_windows(model_dir)
    tickers = sorted(
        set(windows_by_ticker.keys())
        | {p.name.replace("ticker=", "", 1) for p in model_signal_dir.glob("ticker=*")}
        | {p.name.replace("ticker=", "", 1) for p in baseline_signal_dir.glob("ticker=*")}
    )

    non_empty = 0
    for ticker in tickers:
        model_path = model_signal_dir / f"ticker={ticker}" / "data.parquet"
        baseline_path = baseline_signal_dir / f"ticker={ticker}" / "data.parquet"
        model_df = _read_signal_df(model_path)
        baseline_df = _read_signal_df(baseline_path)
        windows = windows_by_ticker.get(ticker, [])

        out_df = build_gated_ticker_df(baseline_df, model_df, windows)

        tdir = output_dir / f"ticker={ticker}"
        tdir.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(tdir / "data.parquet")
        if not out_df.empty:
            non_empty += 1

    print(f"Built gated signal for {len(tickers)} tickers; non_empty={non_empty}")


if __name__ == "__main__":
    main()
