import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import russell1000_close_at_exchange_auction as mod


class _FakeCalendar:
    tz = "Europe/London"

    def schedule(self, start_date=None, end_date=None):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2025-03-07"),
                pd.Timestamp("2025-03-14"),
                pd.Timestamp("2025-12-24"),
            ]
        )
        # Two normal closes at 16:30 London, one early close at 12:30 London.
        market_close = pd.DatetimeIndex(
            [
                pd.Timestamp("2025-03-07 16:30", tz="Europe/London"),
                pd.Timestamp("2025-03-14 16:30", tz="Europe/London"),
                pd.Timestamp("2025-12-24 12:30", tz="Europe/London"),
            ]
        )
        return pd.DataFrame({"market_close": market_close}, index=idx)


def test_compute_exchange_auction_times_uses_local_close_mode(monkeypatch):
    def _fake_get_calendar(_mic):
        return _FakeCalendar()

    monkeypatch.setattr(mod.mcal, "get_calendar", _fake_get_calendar)

    out = mod.compute_exchange_auction_times(
        exchange_mic="XLON",
        offset_str="0min",
        start_date="2025-03-01",
        end_date="2025-12-31",
    )

    # Keep both normal-close dates even though their ET clock times differ
    # around US/UK DST mismatch periods.
    assert len(out) == 2
    assert pd.Timestamp("2025-03-07") in out.index
    assert pd.Timestamp("2025-03-14") in out.index
    assert pd.Timestamp("2025-12-24") not in out.index
