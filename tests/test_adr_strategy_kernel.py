import datetime as dt

import pandas as pd
import polars as pl
import pytest

from adr_strategy_kernel.pipelines.current_covariance import compute_current_covariance
from adr_strategy_kernel.pipelines.fixed_time_mid import extract_daily_fixed_time_mid
from adr_strategy_kernel.pipelines.fixed_time_signal import extract_fixed_time_signal
from adr_strategy_kernel.pipelines.build_aligned_index_prices import identify_misaligned_stocks
from adr_strategy_kernel.pipelines.closing_domestic_prices import get_sh_per_adr
from adr_strategy_kernel.pipelines.only_futures_full_signal import prepare_adr_baseline
from adr_strategy_kernel.pipelines.usd_index_futures import convert_index_futures_to_usd
from adr_strategy_kernel.risk import compute_backtest_covariance


def test_prepare_adr_baseline_uses_asia_override():
    adr_info = pd.DataFrame(
        {
            "id": ["BARC LN Equity", "6758 JT Equity"],
            "adr_ticker": ["BCS", "SONY"],
            "exchange": ["XLON", "XTKS"],
        }
    )
    adr_domestic_close = pd.DataFrame(
        {"BCS": [10.0], "SONY": [20.0]},
        index=["2024-01-02"],
    )
    ord_close_to_usd = pd.DataFrame(
        {"BARC LN Equity": [11.0], "6758 JT Equity": [30.0]},
        index=["2024-01-02"],
    )

    result = prepare_adr_baseline(adr_info, adr_domestic_close, ord_close_to_usd)

    assert result.loc["2024-01-02", "BCS"] == 10.0
    assert result.loc["2024-01-02", "SONY"] == 30.0


def test_get_sh_per_adr_handles_share_reclass_and_splits(tmp_path):
    adr_info_path = tmp_path / "adr_info.csv"
    pd.DataFrame(
        {
            "adr": ["ABC US Equity"],
            "id": ["ABC LN Equity"],
            "sh_per_adr": [2.0],
        }
    ).to_csv(adr_info_path, index=False)

    share_reclass_path = tmp_path / "share_reclass.csv"
    pd.DataFrame(
        {
            "Security ID": ["ABC US Equity"],
            "Effective Date": ["2024-01-04"],
            "Old_Ratio": [1.0],
        }
    ).to_csv(share_reclass_path, index=False)

    splits_path = tmp_path / "all_splits.csv"
    pd.DataFrame(
        {
            "ex_date": ["2024-01-04"],
            "dvd_amt": [2.0],
        },
        index=["ABC US Equity"],
    ).to_csv(splits_path)

    adjusted = get_sh_per_adr(
        start_date="2024-01-02",
        end_date="2024-01-05",
        for_adjusted=True,
        adr_info_path=adr_info_path,
        share_reclass_path=share_reclass_path,
    )
    unadjusted = get_sh_per_adr(
        start_date="2024-01-02",
        end_date="2024-01-05",
        for_adjusted=False,
        adr_info_path=adr_info_path,
        splits_path=splits_path,
    )

    assert adjusted.loc["2024-01-02", "ABC"] == 1.0
    assert adjusted.loc["2024-01-04", "ABC"] == 2.0
    assert unadjusted.loc["2024-01-02", "ABC"] == 4.0
    assert unadjusted.loc["2024-01-04", "ABC"] == 2.0


def test_convert_index_futures_to_usd_writes_notionalized_prices(tmp_path):
    futures_symbols_path = tmp_path / "futures_symbols.csv"
    pd.DataFrame(
        {
            "first_rate_symbol": ["FTUK"],
            "currency": ["GBP"],
        }
    ).to_csv(futures_symbols_path, index=False)

    futures_dir = tmp_path / "futures"
    futures_dir.mkdir()
    (futures_dir / "FTUK_full_1min_continuous_ratio_adjusted.txt").write_text(
        "2024-01-02 09:30:00,100,100,100,100,1\n"
        "2024-01-02 09:31:00,101,101,101,101,1\n",
        encoding="utf-8",
    )

    fx_dir = tmp_path / "fx"
    fx_dir.mkdir()
    (fx_dir / "GBPUSD_full_1min.txt").write_text(
        "20240102,09:30:00,1.25,1.25,1.25,1.25,1\n"
        "20240102,09:31:00,1.25,1.25,1.25,1.25,1\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    convert_index_futures_to_usd(
        futures_symbols_path=futures_symbols_path,
        futures_minute_dir=futures_dir,
        fx_dir=fx_dir,
        output_dir=output_dir,
        symbols=["FTUK"],
    )

    output_file = output_dir / "symbol=FTUK" / "FTUK_close_to_usd_1min.parquet"
    result = pl.read_parquet(output_file)

    assert result["close"].to_list() == [1250.0, 1262.5]
    assert result["fx_rate"].to_list() == [1.25, 1.25]


def test_identify_misaligned_stocks_uses_ten_minute_threshold():
    close_times_df = pd.DataFrame(
        {
            "BLOOMBERG_CLOSE_TIME": ["16:35:00", "16:30:00", "17:00:00"],
        },
        index=["AAA LN Equity", "UKX Index", "BBB LN Equity"],
    )
    stock_to_index = {
        "AAA LN Equity": "UKX",
        "BBB LN Equity": "UKX",
    }

    misaligned = identify_misaligned_stocks(
        ["AAA LN Equity", "BBB LN Equity"],
        close_times_df,
        stock_to_index,
    )

    assert "AAA LN Equity" not in misaligned
    assert "BBB LN Equity" in misaligned


def test_extract_daily_fixed_time_mid_uses_last_quote_before_cutoff(tmp_path):
    nbbo_dir = tmp_path / "nbbo"
    ticker_dir = nbbo_dir / "ticker=ABC"
    ticker_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "nbbo_bid": [9.0, 10.0, 11.0],
            "nbbo_ask": [11.0, 12.0, 13.0],
            "date": ["2024-01-03", "2024-01-03", "2024-01-03"],
        },
        index=pd.DatetimeIndex(
            [
                "2024-01-03 13:35:00-05:00",
                "2024-01-03 13:59:00-05:00",
                "2024-01-03 14:05:00-05:00",
            ],
            name="ts_recv",
        ),
    )
    frame.to_parquet(ticker_dir / "data.parquet")

    result = extract_daily_fixed_time_mid(
        nbbo_dir=nbbo_dir,
        time_to_save=dt.time(14, 0),
        tickers=["ABC"],
        start_date="2024-01-03",
        end_date="2024-01-03",
    )

    assert result.loc[pd.Timestamp("2024-01-03"), "ABC"] == 11.0


def test_extract_fixed_time_signal_uses_last_signal_before_cutoff(tmp_path):
    signal_dir = tmp_path / "signals"
    ticker_dir = signal_dir / "ticker=ABC"
    ticker_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "signal": [1.0, 2.5, 9.0],
            "date": ["2024-01-03", "2024-01-03", "2024-01-03"],
        },
        index=pd.DatetimeIndex(
            [
                "2024-01-03 13:00:00-05:00",
                "2024-01-03 14:00:00-05:00",
                "2024-01-03 14:01:00-05:00",
            ],
            name="timestamp",
        ),
    )
    frame.to_parquet(ticker_dir / "data.parquet")

    result = extract_fixed_time_signal(
        signal_dir=signal_dir,
        tickers=["ABC"],
        time_to_save=dt.time(14, 0),
        start_date="2024-01-03",
        end_date="2024-01-03",
    )

    assert result.loc[pd.Timestamp("2024-01-03"), "ABC"] == 2.5


def test_compute_backtest_covariance_matches_strategy_math():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    adr_signal = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.03],
            "BBB": [0.005, 0.01, 0.015],
        },
        index=dates,
    )
    adr_trade_price = pd.DataFrame(
        {
            "AAA": [90.0, 91.0, 92.0],
            "BBB": [45.0, 46.0, 47.0],
        },
        index=dates,
    )
    adr_close = pd.DataFrame(
        {
            "AAA": [100.0, 100.0, 100.0],
            "BBB": [50.0, 50.0, 50.0],
        },
        index=dates,
    )
    etf_trade_price = pd.DataFrame({"ETF": [80.0, 80.0, 80.0]}, index=dates)
    etf_close = pd.DataFrame({"ETF": [100.0, 100.0, 100.0]}, index=dates)
    hedge_ratios = pd.DataFrame({"AAA": [0.5, 0.5, 0.5], "BBB": [0.25, 0.25, 0.25]}, index=dates)

    covariance, residuals = compute_backtest_covariance(
        trade_date="2024-01-04",
        adr_signal=adr_signal,
        adr_trade_price=adr_trade_price,
        adr_close=adr_close,
        etf_trade_price=etf_trade_price,
        etf_close=etf_close,
        hedge_ratios=hedge_ratios,
        hedge_map={"AAA": "ETF", "BBB": "ETF"},
        vol_lookback=2,
    )

    expected_residuals = pd.DataFrame(
        {
            "AAA": [-0.01, -0.03],
            "BBB": [0.045, 0.02],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    expected_residuals.index.name = "date"
    expected_residuals.columns.name = "ticker"
    pd.testing.assert_frame_equal(residuals, expected_residuals)

    expected_covariance = pd.DataFrame(
        {
            "AAA": [0.0002, 0.00025],
            "BBB": [0.00025, 0.0003125],
        },
        index=["AAA", "BBB"],
    )
    pd.testing.assert_frame_equal(covariance, expected_covariance)


def test_compute_current_covariance_filters_to_trade_symbol_subset(tmp_path):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    def write_csv(path, frame):
        frame.to_csv(path)
        return path

    adr_signal_path = write_csv(
        tmp_path / "fixed_time_signal.csv",
        pd.DataFrame({"AAA": [0.01, 0.02, 0.03], "BBB": [0.005, 0.01, 0.015]}, index=dates),
    )
    adr_trade_path = write_csv(
        tmp_path / "adr_trade.csv",
        pd.DataFrame({"AAA": [90.0, 91.0, 92.0], "BBB": [45.0, 46.0, 47.0]}, index=dates),
    )
    adr_close_path = write_csv(
        tmp_path / "adr_close.csv",
        pd.DataFrame({"AAA": [100.0, 100.0, 100.0], "BBB": [50.0, 50.0, 50.0]}, index=dates),
    )
    etf_trade_path = write_csv(
        tmp_path / "etf_trade.csv",
        pd.DataFrame({"ETF": [80.0, 80.0, 80.0]}, index=dates),
    )
    etf_close_path = write_csv(
        tmp_path / "etf_close.csv",
        pd.DataFrame({"ETF": [100.0, 100.0, 100.0]}, index=dates),
    )
    hedge_ratios_path = write_csv(
        tmp_path / "hedge_ratios.csv",
        pd.DataFrame({"AAA": [0.5, 0.5, 0.5], "BBB": [0.25, 0.25, 0.25]}, index=dates),
    )
    adr_info_path = tmp_path / "adr_info.csv"
    pd.DataFrame(
        {
            "adr": ["AAA US Equity", "BBB US Equity"],
            "market_etf_hedge": ["ETF", "ETF"],
        }
    ).to_csv(adr_info_path, index=False)
    trade_symbols_path = tmp_path / "trade_symbols.csv"
    pd.DataFrame({"ticker": ["AAA"]}).to_csv(trade_symbols_path, index=False)

    result = compute_current_covariance(
        trade_date="2024-01-04",
        adr_signal_path=adr_signal_path,
        adr_trade_price_path=adr_trade_path,
        adr_close_path=adr_close_path,
        etf_trade_price_path=etf_trade_path,
        etf_close_path=etf_close_path,
        hedge_ratios_path=hedge_ratios_path,
        adr_info_path=adr_info_path,
        trade_symbols_csv=trade_symbols_path,
        vol_lookback=2,
    )

    assert list(result.columns) == ["AAA"]
    assert result.loc["AAA", "AAA"] == pytest.approx(0.0002)
