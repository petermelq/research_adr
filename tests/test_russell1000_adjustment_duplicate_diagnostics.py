import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import get_market_business_days


def _prep_group(adj_group: pd.DataFrame) -> pd.DataFrame:
    x = adj_group.copy()
    if "adjustment_factor_operator_type" in x.columns:
        mask = x["adjustment_factor_operator_type"] == 1.0
        x.loc[mask, "adjustment_factor"] = 1.0 / x.loc[mask, "adjustment_factor"]
    return x


def _group_by_date(adj_group: pd.DataFrame) -> pd.DataFrame:
    x = _prep_group(adj_group)
    return (
        x.groupby("adjustment_date", as_index=True)["adjustment_factor"]
        .prod()
        .to_frame("adjustment_factor")
        .sort_index(ascending=False)
    )


def _shift_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    cbday = get_market_business_days("NYSE")
    return pd.DatetimeIndex([pd.to_datetime(i) - cbday for i in idx])


def test_same_day_corporate_actions_are_combined_by_product():
    # Two actions on the same ex-date should both be applied.
    raw = pd.DataFrame(
        {
            "adjustment_date": pd.to_datetime(["2024-01-10", "2024-01-10"]),
            "adjustment_factor": [2.0, 0.8],
            "adjustment_factor_operator_type": [1.0, 2.0],  # divide then multiply
        }
    )

    grouped = _group_by_date(raw)
    assert len(grouped) == 1
    # 1/2.0 * 0.8
    assert grouped.iloc[0]["adjustment_factor"] == 0.4


def test_duplicates_appear_after_shift_not_before():
    # start_date row (2015-01-01) and a real action on 2015-01-02 both shift to 2014-12-31.
    raw = pd.DataFrame(
        {
            "adjustment_date": pd.to_datetime(["2015-01-02"]),
            "adjustment_factor": [1.1],
            "adjustment_factor_operator_type": [2.0],
        }
    )

    grouped = _group_by_date(raw)
    assert not grouped.index.duplicated().any()

    grouped.loc[pd.Timestamp("2015-01-01"), "adjustment_factor"] = 1.0
    shifted = _shift_index(pd.DatetimeIndex(grouped.index))
    assert shifted.duplicated().any()


def test_real_data_first_failure_ticker_a_has_shift_collision():
    df = pd.read_csv("data/processed/russell1000/adjustment_factors.csv", parse_dates=["adjustment_date"])
    df["ticker"] = df["ticker"].str.replace(" US Equity", "", regex=False)
    a = df[df["ticker"] == "A"]

    grouped = _group_by_date(a)
    assert not grouped.index.duplicated().any()

    grouped.loc[pd.Timestamp("2015-01-01"), "adjustment_factor"] = 1.0
    shifted = _shift_index(pd.DatetimeIndex(grouped.index))
    dup = shifted[shifted.duplicated(keep=False)]
    assert len(dup) > 0
    assert "2014-12-31" in set(dup.astype(str))
