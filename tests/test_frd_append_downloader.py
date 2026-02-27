import os
import sys
from pathlib import Path

import pandas as pd


# Import from patched external data_tools repo.
DATA_TOOLS_SRC = "/home/pmalonis/data_tools/src"
if DATA_TOOLS_SRC not in sys.path:
    sys.path.insert(0, DATA_TOOLS_SRC)

from data_tools.cli.frd_download_to_parquet_with_append import (  # noqa: E402
    _read_present_dates_from_none_partition,
    determine_frd_period,
    find_missing_dates,
    get_existing_dates_by_ticker,
)


def _write_ticker_parquet(ticker_dir: Path, df: pd.DataFrame):
    ticker_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ticker_dir / "data.parquet")


def test_read_present_dates_none_partition_from_date_column(tmp_path):
    tdir = tmp_path / "ticker=MSFT"
    idx = pd.DatetimeIndex(
        ["2026-01-02 09:30", "2026-01-02 09:31", "2026-01-06 09:30"],
        tz="America/New_York",
        name="DateTime",
    )
    df = pd.DataFrame(
        {"Open": [1.0, 1.1, 2.0], "Close": [1.1, 1.2, 2.1], "date": ["2026-01-02", "2026-01-02", "2026-01-06"]},
        index=idx,
    )
    _write_ticker_parquet(tdir, df)

    got = _read_present_dates_from_none_partition(str(tdir))
    assert got == ["2026-01-02", "2026-01-06"]


def test_read_present_dates_none_partition_falls_back_to_index(tmp_path):
    tdir = tmp_path / "ticker=NVDA"
    idx = pd.DatetimeIndex(
        ["2026-01-03 09:30", "2026-01-04 09:31", "2026-01-04 09:32"],
        tz="America/New_York",
        name="DateTime",
    )
    # No explicit `date` column, must infer from index.
    df = pd.DataFrame({"Open": [1.0, 1.1, 1.2], "Close": [1.1, 1.2, 1.3]}, index=idx)
    _write_ticker_parquet(tdir, df)

    got = _read_present_dates_from_none_partition(str(tdir))
    assert got == ["2026-01-03", "2026-01-04"]


def test_get_existing_dates_by_ticker_none_partition(tmp_path):
    msft = tmp_path / "ticker=MSFT"
    nvda = tmp_path / "ticker=NVDA"

    msft_idx = pd.DatetimeIndex(["2026-01-02 09:30", "2026-01-02 09:31"], tz="America/New_York", name="DateTime")
    nvda_idx = pd.DatetimeIndex(["2026-01-05 09:30"], tz="America/New_York", name="DateTime")

    _write_ticker_parquet(msft, pd.DataFrame({"Close": [1.0, 1.1], "date": ["2026-01-02", "2026-01-02"]}, index=msft_idx))
    _write_ticker_parquet(nvda, pd.DataFrame({"Close": [2.0]}, index=nvda_idx))

    got = get_existing_dates_by_ticker(str(tmp_path), partition_period="none")
    assert got["MSFT"] == ["2026-01-02"]
    assert got["NVDA"] == ["2026-01-05"]


def test_find_missing_dates():
    expected = ["2026-01-02", "2026-01-05", "2026-01-06"]
    present = ["2026-01-02"]
    assert find_missing_dates(expected, present) == ["2026-01-05", "2026-01-06"]


def test_determine_period_uses_requested_end_date():
    # If end_date is historical, period selection should be relative to it.
    # Missing a day right before end_date -> day
    assert determine_frd_period("2026-02-12", end_date="2026-02-13") == "day"
    # Missing within same week -> week
    assert determine_frd_period("2026-02-09", end_date="2026-02-13") == "week"
    # Older than 28 days from end_date -> full
    assert determine_frd_period("2025-12-15", end_date="2026-02-13") == "full"
