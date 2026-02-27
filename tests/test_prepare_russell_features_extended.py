import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from prepare_russell_features_extended import (
    _compute_us_open_close_for_ticker,
    build_feature_matrix,
    load_ticker_set_from_csv,
    resolve_ny_close_target_adrs,
    select_feature_columns_by_coverage,
    validate_feature_outputs,
)


def test_select_feature_columns_by_coverage_drops_sparse_recent_and_total():
    dates = pd.bdate_range("2025-01-01", periods=100)
    df = pd.DataFrame(index=dates)
    rng = np.random.default_rng(0)

    df["GOOD"] = rng.normal(size=len(dates))
    df["SPARSE_RECENT"] = rng.normal(size=len(dates))
    # Only 10/20 non-null in recent window -> should fail recent coverage threshold.
    df.loc[dates[-20:-10], "SPARSE_RECENT"] = np.nan
    df.loc[dates[-10:], "SPARSE_RECENT"] = np.nan

    df["SPARSE_TOTAL"] = np.nan
    df.loc[dates[:20], "SPARSE_TOTAL"] = rng.normal(size=20)

    df["RECENT_OK"] = rng.normal(size=len(dates))
    # 14/20 non-null in recent window and high total coverage -> should pass.
    df.loc[dates[-20:-14], "RECENT_OK"] = np.nan

    keep = select_feature_columns_by_coverage(
        df,
        dates,
        recent_days=20,
        min_total_coverage=0.40,
        min_recent_coverage=0.60,
        min_recent_obs=10,
    )

    assert "GOOD" in keep
    assert "RECENT_OK" in keep
    assert "SPARSE_RECENT" not in keep
    assert "SPARSE_TOTAL" not in keep


def test_build_feature_matrix_keeps_target_non_imputed_and_filters_sparse_features():
    dates = pd.bdate_range("2025-01-01", periods=80)
    rng = np.random.default_rng(1)

    ordinary_residuals = pd.Series(rng.normal(size=len(dates)), index=dates, name="ordinary_residual")
    # Target NaN should be dropped, not replaced with 0.
    ordinary_residuals.loc[dates[[3, 7]]] = np.nan

    russell_residuals = pd.DataFrame(index=dates)
    russell_residuals["GOOD"] = rng.normal(size=len(dates))
    russell_residuals["GOOD_WITH_HOLES"] = rng.normal(size=len(dates))
    russell_residuals.loc[dates[[10, 11, 12]], "GOOD_WITH_HOLES"] = np.nan
    russell_residuals["BAD_RECENT"] = rng.normal(size=len(dates))
    # Make the recent window sparse enough to fail default recent-coverage filter.
    russell_residuals.loc[dates[-40:], "BAD_RECENT"] = np.nan

    features = build_feature_matrix(ordinary_residuals, russell_residuals, min_feature_count=2)
    assert features is not None

    # NaN targets are dropped, never zero-imputed.
    assert features["ordinary_residual"].isna().sum() == 0
    assert dates[3] not in features.index
    assert dates[7] not in features.index

    # Sparse recent column should be removed.
    assert "russell_BAD_RECENT" not in features.columns
    assert "russell_GOOD" in features.columns
    assert "russell_GOOD_WITH_HOLES" in features.columns

    # Remaining feature holes should be imputed to 0.
    assert features["russell_GOOD_WITH_HOLES"].isna().sum() == 0


def test_build_feature_matrix_returns_none_when_not_enough_features():
    dates = pd.bdate_range("2025-01-01", periods=40)
    ordinary_residuals = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
    russell_residuals = pd.DataFrame({"ONLY_ONE": np.linspace(0.0, 0.02, len(dates))}, index=dates)

    out = build_feature_matrix(ordinary_residuals, russell_residuals, min_feature_count=2)
    assert out is None


def test_compute_us_open_close_supports_datetime_column_schema(tmp_path):
    ticker = "ABC"
    out_dir = tmp_path / "russell"
    tdir = out_dir / f"ticker={ticker}"
    tdir.mkdir(parents=True, exist_ok=True)

    dt = pd.to_datetime(
        [
            "2026-01-05 09:30:00",
            "2026-01-05 16:00:00",
            "2026-01-06 09:30:00",
            "2026-01-06 16:00:00",
        ]
    ).tz_localize("America/New_York")
    df = pd.DataFrame({"DateTime": dt, "Close": [100.0, 101.0, 200.0, 198.0]})
    df.to_parquet(tdir / "data.parquet")

    _, ret = _compute_us_open_close_for_ticker((str(out_dir), ticker, "2026-01-01", "2026-01-31"))
    assert ret is not None
    assert len(ret) == 2
    assert abs(ret.iloc[0] - 0.01) < 1e-12
    assert abs(ret.iloc[1] - (-0.01)) < 1e-12


def test_validate_feature_outputs_detects_missing_and_schema_errors(tmp_path):
    out_dir = tmp_path / "features_extended"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.bdate_range("2026-01-01", periods=3)
    good = pd.DataFrame(
        {
            "ordinary_residual": [0.1, 0.2, 0.3],
            "russell_A": [1.0, 2.0, 3.0],
            "russell_B": [0.1, 0.2, 0.3],
        },
        index=idx,
    )
    good.to_parquet(out_dir / "AAA.parquet")

    bad = pd.DataFrame({"russell_A": [1.0, 2.0, 3.0], "russell_B": [1.0, 2.0, 3.0]}, index=idx)
    bad.to_parquet(out_dir / "BBB.parquet")

    # Missing ticker CCC and broken BBB should trigger failure.
    try:
        validate_feature_outputs(
            output_dir=out_dir,
            expected_adrs={"AAA", "BBB", "CCC"},
            min_feature_count=2,
            min_success_ratio=1.0,
        )
        assert False, "Expected validation failure"
    except RuntimeError as e:
        msg = str(e)
        assert "validation failed" in msg.lower() or "schema validation failed" in msg.lower()


def test_resolve_ny_close_target_adrs_default_and_override():
    exchange_to_tickers = {
        "XETR": [("SAP", "SAP", "SX5E")],
        "XTKS": [("7203 JP Equity", "TM", "NKY"), ("6758 JP Equity", "SONY", "NKY")],
        "XASX": [("BHP AU Equity", "BHP", "AS51")],
    }
    default_set = resolve_ny_close_target_adrs(exchange_to_tickers)
    assert default_set == {"TM", "SONY", "BHP"}

    override_set = resolve_ny_close_target_adrs(exchange_to_tickers, explicit_tickers={"SAP"})
    assert override_set == {"SAP"}


def test_load_ticker_set_from_csv_column_fallback(tmp_path):
    fp = tmp_path / "tickers.csv"
    pd.DataFrame({"custom": ["AAA", "BBB", None, "  CCC  "]}).to_csv(fp, index=False)
    out = load_ticker_set_from_csv(fp, column="ticker")
    assert out == {"AAA", "BBB", "CCC"}
