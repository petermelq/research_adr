import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_time_list(value: str):
    times = [x.strip() for x in value.split(",") if x.strip()]
    for t in times:
        hh, mm = t.split(":")
        int(hh)
        int(mm)
    return times


def load_signal_at_times(signal_dir: Path, ticker: str, times: list[str]) -> dict[str, pd.Series]:
    path = signal_dir / f"ticker={ticker}" / "data.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["signal"])
    out = {}
    for t in times:
        hh, mm = map(int, t.split(":"))
        mask = (df.index.hour == hh) & (df.index.minute == mm)
        s = df.loc[mask, "signal"]
        if len(s) == 0:
            continue
        idx = s.index.tz_localize(None).normalize()
        out[t] = pd.Series(s.values, index=idx).groupby(level=0).first().astype(float)
    return out


def load_entry_mid_times(nbbo_path: Path, times: list[str], close_window=("15:59", "16:00")) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(nbbo_path, columns=["nbbo_bid", "nbbo_ask"])
    mid = ((df["nbbo_bid"] + df["nbbo_ask"]) / 2.0).astype(float)
    tmp = pd.DataFrame({"mid": mid}, index=df.index)
    tmp["date"] = tmp.index.tz_localize(None).normalize()
    all_dates = sorted(tmp["date"].unique())
    out = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
    for t in times:
        s = tmp.between_time(t, t)["mid"]
        if len(s) == 0:
            out[t] = np.nan
            continue
        idx = s.index.tz_localize(None).normalize()
        ser = pd.Series(s.values, index=idx).groupby(level=0).first()
        out[t] = ser.reindex(out.index)
    close_s = tmp.between_time(close_window[0], close_window[1])["mid"]
    close_idx = close_s.index.tz_localize(None).normalize()
    close_ser = pd.Series(close_s.values, index=close_idx).groupby(level=0).last().reindex(out.index)
    return out, close_ser.astype(float)


def fit_convex_blend(x_base: np.ndarray, x_model: np.ndarray, y: np.ndarray):
    diff = x_model - x_base
    denom = float(np.dot(diff, diff))
    if not np.isfinite(denom) or denom <= 1e-12:
        w = 0.5
    else:
        w = float(np.dot(y - x_base, diff) / denom)
    w = float(np.clip(w, 0.0, 1.0))
    pred = (1.0 - w) * x_base + w * x_model
    b = float(np.mean(y - pred))
    return w, b


def train_monthly_moe(
    panel: pd.DataFrame,
    train_times: list[str],
    output_times: list[str],
    lookback_months: int,
) -> pd.DataFrame:
    panel = panel.sort_values(["date", "entry_time"]).copy()
    panel["month"] = panel["date"].dt.to_period("M")
    months = sorted(panel["month"].dropna().unique())

    # Map non-training output times to the nearest trained afternoon bucket.
    output_to_train = {
        "13:00": "13:30",
        "13:30": "13:30",
        "14:00": "14:00",
        "14:30": "14:30",
        "15:00": "14:30",
        "15:30": "14:30",
    }
    for t in output_times:
        output_to_train.setdefault(t, train_times[min(range(len(train_times)), key=lambda i: abs(int(t[:2]) * 60 + int(t[3:]) - (int(train_times[i][:2]) * 60 + int(train_times[i][3:]))))])

    weight_rows = []
    pred_rows = []
    for month in months:
        month_start = month.to_timestamp(how="start")
        train_start = month_start - pd.DateOffset(months=lookback_months)
        train_mask = (
            (panel["date"] >= train_start)
            & (panel["date"] < month_start)
            & (panel["entry_time"].isin(train_times))
        )
        train_df = panel.loc[train_mask].copy()

        month_df = panel.loc[panel["month"] == month].copy()
        if month_df.empty:
            continue

        # Fit one convex blend per training time bucket.
        w_map = {}
        b_map = {}
        for t in train_times:
            td = train_df.loc[train_df["entry_time"] == t]
            td = td.dropna(subset=["base_signal", "model_signal", "target_return"])
            if len(td) < 20:
                w_map[t] = 0.5
                b_map[t] = 0.0
            else:
                w_t, b_t = fit_convex_blend(
                    td["base_signal"].to_numpy(dtype=float),
                    td["model_signal"].to_numpy(dtype=float),
                    td["target_return"].to_numpy(dtype=float),
                )
                w_map[t] = w_t
                b_map[t] = b_t
            weight_rows.append(
                {
                    "month": str(month),
                    "train_time": t,
                    "n_train": int(len(td)),
                    "w_model": float(w_map[t]),
                    "w_base": float(1.0 - w_map[t]),
                    "bias": float(b_map[t]),
                }
            )

        for _, r in month_df.iterrows():
            et = r["entry_time"]
            train_bucket = output_to_train.get(et, et)
            w = float(w_map.get(train_bucket, 0.5))
            b = float(b_map.get(train_bucket, 0.0))
            base = float(r["base_signal"])
            model = float(r["model_signal"])
            pred = (1.0 - w) * base + w * model + b
            pred_rows.append(
                {
                    "DateTime": r["DateTime"],
                    "date": r["date"],
                    "entry_time": et,
                    "signal": pred,
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    weights_df = pd.DataFrame(weight_rows)
    return pred_df, weights_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-signal-dir", default="data/processed/index_russell_pcr_signal_30m")
    p.add_argument("--baseline-signal-dir", default="data/processed/futures_only_signal")
    p.add_argument("--output-dir", default="data/processed/index_russell_moe_signal_30m")
    p.add_argument("--weights-out", default="data/processed/index_russell_moe_signal_30m_weights.csv")
    p.add_argument("--lookback-months", type=int, default=6)
    p.add_argument("--train-times", default="13:30,14:00,14:30")
    p.add_argument("--output-times", default="13:00,13:30,14:00,14:30,15:00,15:30")
    p.add_argument("--min-output-date", default="2025-09-01")
    p.add_argument("--close-source", choices=["nbbo", "daily"], default="daily")
    args = p.parse_args()

    model_dir = Path(args.model_signal_dir)
    base_dir = Path(args.baseline_signal_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_out = Path(args.weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    train_times = parse_time_list(args.train_times)
    output_times = parse_time_list(args.output_times)
    min_output_date = pd.Timestamp(args.min_output_date)

    adr_info = pd.read_csv("data/raw/adr_info.csv")
    adr_info["ticker"] = adr_info["adr"].str.replace(" US Equity", "", regex=False)
    tickers = sorted(adr_info["ticker"].unique().tolist())

    close_df = None
    if args.close_source == "daily":
        close_df = pd.read_csv("data/raw/adrs/adr_PX_LAST_adjust_none.csv", index_col=0, parse_dates=True)
        close_df.index = pd.to_datetime(close_df.index).normalize()

    all_weights = []
    written = 0
    for ticker in tickers:
        nbbo_path = Path("data/raw/adrs/bbo-1m/nbbo") / f"ticker={ticker}" / "data.parquet"
        if not nbbo_path.exists():
            continue

        model_map = load_signal_at_times(model_dir, ticker, output_times)
        base_map = load_signal_at_times(base_dir, ticker, output_times)
        if not model_map or not base_map:
            continue

        mid_times, nbbo_close = load_entry_mid_times(nbbo_path, output_times)
        if args.close_source == "daily":
            if ticker not in close_df.columns:
                continue
            ticker_close = close_df[ticker].dropna().astype(float)
        else:
            ticker_close = nbbo_close.dropna().astype(float)
        common_dates = mid_times.index.intersection(ticker_close.index)
        if len(common_dates) == 0:
            continue

        # Build panel rows across output times.
        rows = []
        for et in output_times:
            if et not in model_map or et not in base_map or et not in mid_times.columns:
                continue
            m = model_map[et]
            b = base_map[et]
            entry_mid = mid_times[et].reindex(common_dates).astype(float)
            close_px = ticker_close.reindex(common_dates).astype(float)
            y = (close_px / entry_mid) - 1.0

            idx = common_dates.intersection(m.index).intersection(b.index).intersection(y.dropna().index)
            if len(idx) == 0:
                continue
            for d in idx:
                rows.append(
                    {
                        "date": d,
                        "entry_time": et,
                        "model_signal": float(m.loc[d]),
                        "base_signal": float(b.loc[d]),
                        "target_return": float(y.loc[d]),
                        "DateTime": pd.Timestamp(f"{d.date()} {et}"),
                    }
                )

        if not rows:
            continue
        panel = pd.DataFrame(rows)
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_signal", "base_signal", "target_return"])
        if panel.empty:
            continue

        pred_df, w_df = train_monthly_moe(
            panel=panel,
            train_times=train_times,
            output_times=output_times,
            lookback_months=int(args.lookback_months),
        )
        if pred_df.empty:
            continue

        pred_df = pred_df[pred_df["date"] >= min_output_date].copy()
        if pred_df.empty:
            continue
        pred_df["DateTime"] = pd.to_datetime(pred_df["DateTime"]).dt.tz_localize("America/New_York")
        pred_df = pred_df.sort_values("DateTime").set_index("DateTime")
        out = pred_df[["signal", "date"]].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

        ticker_out_dir = out_dir / f"ticker={ticker}"
        ticker_out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(ticker_out_dir / "data.parquet")
        written += 1

        if not w_df.empty:
            w_df["ticker"] = ticker
            all_weights.append(w_df)

    if all_weights:
        weights = pd.concat(all_weights, ignore_index=True)
        weights.to_csv(weights_out, index=False)
    else:
        pd.DataFrame(columns=["ticker", "month", "train_time", "n_train", "w_model", "w_base", "bias"]).to_csv(weights_out, index=False)

    print(f"Wrote blended MoE signals for {written} tickers -> {out_dir}")
    print(f"Saved weights -> {weights_out}")


if __name__ == "__main__":
    main()
