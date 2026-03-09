from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yaml

ADR_SUFFIX = " US Equity"
ASIA_EXCHANGES = {"XTKS", "XASX", "XHKG", "XSES", "XSHG", "XSHE"}


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(*parts: str, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    return root.joinpath(*parts)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def strip_us_equity_suffix(value: Any):
    if isinstance(value, pd.Series):
        return value.astype(str).str.replace(ADR_SUFFIX, "", regex=False)
    if isinstance(value, pd.Index):
        return pd.Index([strip_us_equity_suffix(item) for item in value])
    if value is None:
        return None
    return str(value).replace(ADR_SUFFIX, "")


def normalize_currency(currency: Any) -> str | None:
    if not isinstance(currency, str):
        return None
    cleaned = currency.strip()
    if cleaned == "" or cleaned.lower() in {"nan", "none", "null"}:
        return None
    if cleaned == "GBp":
        return "GBP"
    return cleaned.upper()


def is_usd_currency(currency: Any) -> bool:
    return normalize_currency(currency) == "USD"


def load_params(
    params_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(params_path) if params_path is not None else resolve_repo_path("params.yaml", repo_root=repo_root)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def load_adr_info(
    adr_info_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(adr_info_path) if adr_info_path is not None else resolve_repo_path(
        "data", "raw", "adr_info.csv", repo_root=repo_root
    )
    adr_info = pd.read_csv(path).copy()
    if "adr" in adr_info.columns:
        adr_info["adr_ticker"] = strip_us_equity_suffix(adr_info["adr"])
    if "index_future_bbg" in adr_info.columns:
        adr_info["index_future_bbg"] = adr_info["index_future_bbg"].astype(str).str.strip()
    return adr_info


def load_close_offsets(close_offsets_path: str | Path) -> dict[str, pd.Timedelta]:
    offsets_df = pd.read_csv(close_offsets_path)
    return {
        row["exchange_mic"]: pd.Timedelta(str(row["offset"]))
        for _, row in offsets_df.iterrows()
    }


def get_market_business_days(calendar: str = "NYSE") -> pd.offsets.CustomBusinessDay:
    cal = mcal.get_calendar(calendar)
    holidays = cal.holidays().holidays
    if calendar in ("XNYS", "NYSE"):
        holidays += (np.datetime64("2025-01-09"),)
    return pd.offsets.CustomBusinessDay(holidays=holidays)
