from pathlib import Path

import numpy as np
import pandas as pd
import pytest


FEATURE_DIR = Path("data/processed/models/with_us_stocks/features_extended")
ADR_INFO_PATH = Path("data/raw/adr_info.csv")
ASIA_EXCHANGES = {"XTKS", "XASX", "XHKG", "XSES", "XSHG", "XSHE"}


@pytest.fixture(scope="module")
def feature_quality_snapshot():
    if not FEATURE_DIR.exists():
        pytest.skip(f"Feature directory missing: {FEATURE_DIR}")

    files = sorted(FEATURE_DIR.glob("*.parquet"))
    if not files:
        pytest.skip(f"No feature files found in: {FEATURE_DIR}")

    adr_info = pd.read_csv(ADR_INFO_PATH)
    adr_info["adr_ticker"] = adr_info["adr"].str.replace(" US Equity", "", regex=False)
    adr_to_exchange = dict(zip(adr_info["adr_ticker"], adr_info["exchange"]))

    feature_sets = {}
    rows = []
    union_features = set()

    for fp in files:
        ticker = fp.stem
        df = pd.read_parquet(fp)
        russell_cols = [c for c in df.columns if c.startswith("russell_")]

        if not russell_cols:
            rows.append(
                {
                    "ticker": ticker,
                    "exchange": adr_to_exchange.get(ticker, "UNKNOWN"),
                    "n_rows": int(len(df)),
                    "n_russell": 0,
                    "null_frac": 1.0,
                    "zero_frac": 1.0,
                    "ordinary_zero_frac": 1.0,
                }
            )
            feature_sets[ticker] = set()
            continue

        rarr = df[russell_cols].to_numpy(dtype=float, copy=False)
        null_frac = float(np.isnan(rarr).mean())
        zero_frac = float((rarr == 0).mean())
        ordinary_zero_frac = (
            float((df["ordinary_residual"] == 0).mean())
            if "ordinary_residual" in df.columns
            else 1.0
        )

        rset = set(russell_cols)
        feature_sets[ticker] = rset
        union_features.update(rset)
        rows.append(
            {
                "ticker": ticker,
                "exchange": adr_to_exchange.get(ticker, "UNKNOWN"),
                "n_rows": int(len(df)),
                "n_russell": int(len(rset)),
                "null_frac": null_frac,
                "zero_frac": zero_frac,
                "ordinary_zero_frac": ordinary_zero_frac,
            }
        )

    meta = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    return {
        "meta": meta,
        "feature_sets": feature_sets,
        "union_features": union_features,
    }


def test_features_not_missing_substantial_russell_coverage(feature_quality_snapshot):
    snapshot = feature_quality_snapshot
    meta = snapshot["meta"]
    feature_sets = snapshot["feature_sets"]
    union_features = snapshot["union_features"]

    assert len(union_features) >= 1000, "Russell feature universe unexpectedly small."

    min_ratio = 0.80
    failures = []
    for _, row in meta.iterrows():
        ticker = row["ticker"]
        ratio = len(feature_sets[ticker]) / len(union_features) if union_features else 0.0
        if ratio < min_ratio:
            failures.append((ticker, row["exchange"], len(feature_sets[ticker]), ratio))

    assert not failures, (
        "Tickers with substantial Russell feature loss "
        f"(threshold={min_ratio:.0%} of union): {failures[:10]}"
    )

    major = {"russell_MSFT", "russell_NVDA", "russell_AAPL", "russell_AMZN"}
    major_missing = []
    for exchange, ex_df in meta.groupby("exchange"):
        if exchange in ASIA_EXCHANGES or len(ex_df) == 0:
            continue
        max_rows = int(ex_df["n_rows"].max())
        long_hist = ex_df[ex_df["n_rows"] >= int(0.90 * max_rows)]
        for ticker in long_hist["ticker"]:
            missing = sorted(major - feature_sets[ticker])
            if missing:
                major_missing.append((ticker, exchange, missing))

    assert not major_missing, (
        "Major Russell names missing from non-Asia long-history ordinaries: "
        f"{major_missing[:10]}"
    )


def test_missing_features_consistent_between_comparable_ordinaries(feature_quality_snapshot):
    snapshot = feature_quality_snapshot
    meta = snapshot["meta"]
    feature_sets = snapshot["feature_sets"]
    union_features = snapshot["union_features"]

    consistency_failures = []
    for exchange, ex_df in meta.groupby("exchange"):
        if len(ex_df) < 2:
            continue
        max_rows = int(ex_df["n_rows"].max())
        # Compare only ordinaries with similar history depth to avoid
        # expected differences from shorter listing history.
        anchors = ex_df[ex_df["n_rows"] >= int(0.90 * max_rows)]["ticker"].tolist()
        if len(anchors) < 2:
            continue

        anchor_missing = [union_features - feature_sets[t] for t in anchors]
        base = anchor_missing[0]
        for ticker, miss in zip(anchors[1:], anchor_missing[1:]):
            union_n = len(base | miss)
            jaccard = 1.0 if union_n == 0 else len(base & miss) / union_n
            if jaccard < 0.98:
                consistency_failures.append((exchange, ticker, round(jaccard, 4)))

    assert not consistency_failures, (
        "Inconsistent missing-feature signatures for comparable ordinaries: "
        f"{consistency_failures[:10]}"
    )


def test_feature_files_not_null_or_zero_heavy(feature_quality_snapshot):
    meta = feature_quality_snapshot["meta"]

    null_fail = meta[meta["null_frac"] > 0.005][["ticker", "null_frac"]]
    zero_fail = meta[meta["zero_frac"] > 0.40][["ticker", "zero_frac"]]
    ordinary_zero_fail = meta[meta["ordinary_zero_frac"] > 0.05][["ticker", "ordinary_zero_frac"]]

    assert null_fail.empty, (
        "Feature files with substantial Russell NaN mass (>0.5%): "
        f"{null_fail.to_dict(orient='records')[:10]}"
    )
    assert zero_fail.empty, (
        "Feature files with substantial Russell zero mass (>40%): "
        f"{zero_fail.to_dict(orient='records')[:10]}"
    )
    assert ordinary_zero_fail.empty, (
        "Feature files with substantial zero ordinary residual mass (>5%): "
        f"{ordinary_zero_fail.to_dict(orient='records')[:10]}"
    )
