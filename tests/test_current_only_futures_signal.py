import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "live_trading" / "current_only_futures_signal.py"
SPEC = importlib.util.spec_from_file_location("current_only_futures_signal", MODULE_PATH)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def test_select_price_at_or_before_returns_last_eligible_bar():
    series = pd.Series(
        [100.0, 101.0, 103.0],
        index=pd.DatetimeIndex(
            [
                "2026-03-10 10:00:00-04:00",
                "2026-03-10 10:05:00-04:00",
                "2026-03-10 10:10:00-04:00",
            ]
        ),
        name="price",
    )

    cutoff = pd.Timestamp("2026-03-10 10:06:00", tz="America/New_York")
    assert mod.select_price_at_or_before(series, cutoff) == 101.0


def test_compute_live_signals_uses_cached_baselines_and_current_snapshots(monkeypatch):
    baseline_df = pd.DataFrame(
        {
            "adr_ticker": ["AZN", "SAP"],
            "exchange": ["XLON", "XETR"],
            "beta": [0.8, 1.2],
            "close_time": pd.to_datetime(
                ["2026-03-10 11:36:00-04:00", "2026-03-10 11:36:00-04:00"]
            ),
            "adr_bbg": ["AZN US Equity", "SAP US Equity"],
            "baseline_adr": [70.0, 200.0],
            "future_bbg": ["ZH6 Index", "VGH6 Index"],
            "future_currency": ["GBP", "EUR"],
            "baseline_future_native": [9000.0, 5000.0],
            "fx_ticker": ["GBPUSD Curncy", "EURUSD Curncy"],
            "baseline_fx": [1.30, 1.08],
        }
    )

    snapshots = {
        mod.FIELD_PX_LAST: {
            "AZN US Equity": 71.4,
            "SAP US Equity": 199.0,
            "ZH6 Index": 9090.0,
            "VGH6 Index": 5050.0,
            "GBPUSD Curncy": 1.31,
            "EURUSD Curncy": 1.09,
        }
    }

    def fake_fetch(tickers, field, timeout_ms):
        return pd.Series({ticker: snapshots[field][ticker] for ticker in tickers})

    monkeypatch.setattr(mod, "fetch_bdp_snapshot", fake_fetch)
    monkeypatch.setattr(mod, "ny_now", lambda: pd.Timestamp("2026-03-10 13:15:00", tz="America/New_York"))

    result = mod.compute_live_signals(baseline_df=baseline_df, timeout_ms=5000)

    azn = result.set_index("adr_ticker").loc["AZN"]
    expected_fut_ret = (9090.0 * 1.31) / (9000.0 * 1.30) - 1.0
    expected_adr_ret = 71.4 / 70.0 - 1.0
    expected_signal = expected_fut_ret * 0.8 - expected_adr_ret

    assert abs(azn["future_return"] - expected_fut_ret) < 1e-12
    assert abs(azn["adr_return"] - expected_adr_ret) < 1e-12
    assert abs(azn["signal"] - expected_signal) < 1e-12
