from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def latest_text_date(path: Path, fmt: str) -> str:
    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line.strip()
    if not last_line:
        raise ValueError(f"{path} is empty")
    return str(pd.to_datetime(last_line.split(",")[0], format=fmt).date())


def latest_parquet_date(path: Path) -> str:
    df = pd.read_parquet(path, columns=["date"])
    return str(pd.to_datetime(df["date"].astype(str)).max().date())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that imported raw inputs cover the trade date.")
    parser.add_argument("--context-dir", required=True)
    parser.add_argument("--imports-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    context_dir = Path(args.context_dir)
    imports_root = Path(args.imports_root)
    runtime = json.loads((context_dir / "runtime_config.json").read_text(encoding="utf-8"))
    trade_date = str(pd.Timestamp(runtime["trade_date"]).date())

    checks: list[dict[str, str]] = []
    failures: list[str] = []

    adr_tickers = pd.read_csv(context_dir / "adr_tickers.csv")["ticker"].tolist()
    etf_tickers = pd.read_csv(context_dir / "market_etf_tickers.csv")["ticker"].tolist()
    futures_symbols = pd.read_csv(context_dir / "required_futures.csv")["first_rate_symbol"].tolist()
    fx_currencies = pd.read_csv(context_dir / "required_fx.csv")["currency"].tolist()

    for ticker in adr_tickers:
        data_path = imports_root / "data" / "raw" / "adrs" / "bbo-1m" / "nbbo" / f"ticker={ticker}" / "data.parquet"
        latest = latest_parquet_date(data_path)
        checks.append({"kind": "adr_nbbo", "key": ticker, "latest_date": latest})
        if latest < trade_date:
            failures.append(f"ADR NBBO for {ticker} ends at {latest}, before trade date {trade_date}")

    for ticker in etf_tickers:
        data_path = imports_root / "data" / "raw" / "etfs" / "market" / "bbo-1m" / "nbbo" / f"ticker={ticker}" / "data.parquet"
        latest = latest_parquet_date(data_path)
        checks.append({"kind": "market_etf_nbbo", "key": ticker, "latest_date": latest})
        if latest < trade_date:
            failures.append(f"Market ETF NBBO for {ticker} ends at {latest}, before trade date {trade_date}")

    for symbol in futures_symbols:
        data_path = imports_root / "data" / "raw" / "futures" / "minute_bars" / f"{symbol}_full_1min_continuous_ratio_adjusted.txt"
        latest = latest_text_date(data_path, "%Y-%m-%d %H:%M:%S")
        checks.append({"kind": "futures_minute", "key": symbol, "latest_date": latest})
        if latest < trade_date:
            failures.append(f"Futures minute data for {symbol} ends at {latest}, before trade date {trade_date}")

    for currency in fx_currencies:
        data_path = imports_root / "data" / "raw" / "currencies" / "minute_bars" / f"{currency}USD_full_1min.txt"
        latest = latest_text_date(data_path, "%Y%m%d")
        checks.append({"kind": "fx_minute", "key": currency, "latest_date": latest})
        if latest < trade_date:
            failures.append(f"FX minute data for {currency}USD ends at {latest}, before trade date {trade_date}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"trade_date": trade_date, "checks": checks}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError("\n".join(failures))


if __name__ == "__main__":
    main()
