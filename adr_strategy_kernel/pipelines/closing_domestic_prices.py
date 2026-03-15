from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from adr_strategy_kernel.common import (
    ensure_parent_dir,
    get_market_business_days,
    load_adr_info,
    resolve_repo_path,
)

SUPPORTED_FX_CURRENCIES = {"GBp", "GBP", "EUR", "JPY", "AUD", "NOK", "SEK", "DKK", "CHF", "HKD"}


def get_sh_per_adr(
    start_date: str,
    end_date: str,
    for_adjusted: bool,
    adr_info_path: str | Path | None = None,
    share_reclass_path: str | Path | None = None,
    splits_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    adr_info_file = Path(adr_info_path) if adr_info_path is not None else resolve_repo_path(
        "data", "raw", "adr_info.csv", repo_root=repo_root
    )
    adr_info = pd.read_csv(adr_info_file)
    cbday = get_market_business_days("NYSE")
    index = pd.date_range(start=start_date, end=end_date, freq=cbday)
    sh_per_adr = pd.DataFrame(
        {
            ticker: [ratio] * len(index)
            for ticker, ratio in adr_info.set_index("adr")["sh_per_adr"].to_dict().items()
        },
        index=index,
    )

    if for_adjusted:
        reclass_file = Path(share_reclass_path) if share_reclass_path is not None else resolve_repo_path(
            "data", "raw", "adrs", "share_reclass.csv", repo_root=repo_root
        )
        reclass_df = pd.read_csv(reclass_file)
        reclass_df["Effective Date"] = pd.to_datetime(reclass_df["Effective Date"])
        reclass_df = reclass_df[
            (reclass_df["Effective Date"] >= pd.to_datetime(start_date))
            & (reclass_df["Effective Date"] <= pd.to_datetime(end_date))
        ].sort_values("Effective Date", ascending=False)

        for _, row in reclass_df.iterrows():
            ticker = row["Security ID"]
            old_ratio = row["Old_Ratio"]
            sh_per_adr.loc[sh_per_adr.index < row["Effective Date"], ticker] = old_ratio
    else:
        splits_file = Path(splits_path) if splits_path is not None else resolve_repo_path(
            "data", "raw", "all_splits.csv", repo_root=repo_root
        )
        split_df = pd.read_csv(splits_file, index_col=0, parse_dates=["ex_date"]).sort_values("ex_date", ascending=False)
        split_df = split_df[
            (split_df["ex_date"] >= pd.to_datetime(start_date))
            & (split_df["ex_date"] <= pd.to_datetime(end_date))
        ]
        for ticker, row in split_df.iterrows():
            ratio = row["dvd_amt"]
            if ticker in adr_info["adr"].values:
                sh_per_adr.loc[sh_per_adr.index < row["ex_date"], ticker] = (
                    sh_per_adr.loc[sh_per_adr.index < row["ex_date"], ticker] * ratio
                )
            elif ticker in adr_info["id"].values:
                adr_ticker = adr_info.loc[adr_info["id"] == ticker, "adr"].values[0]
                sh_per_adr.loc[sh_per_adr.index < row["ex_date"], adr_ticker] = (
                    sh_per_adr.loc[sh_per_adr.index < row["ex_date"], adr_ticker] / ratio
                )

    sh_per_adr.columns = [column.replace(" US Equity", "") for column in sh_per_adr.columns]
    return sh_per_adr


def load_fx_prices(
    fx_dir: str | Path,
    currencies: list[str],
) -> pd.DataFrame:
    currency_map = {"GBp": "GBP"}
    all_fx_data: list[pd.DataFrame] = []
    for currency in currencies:
        source_currency = currency_map.get(currency, currency)
        fx_file = Path(fx_dir) / f"{source_currency}USD_full_1min.txt"
        fx_df = pd.read_csv(
            fx_file,
            header=None,
            index_col=None,
            names=["date", "time", "open", "high", "low", "close", "volume"],
        )
        fx_df["timestamp"] = pd.to_datetime(
            fx_df["date"].astype(str) + " " + fx_df["time"].astype(str)
        ).dt.tz_localize("America/New_York")

        if currency == "GBp":
            fx_df[["open", "high", "low", "close"]] = fx_df[["open", "high", "low", "close"]] / 100
        fx_df["currency"] = currency
        all_fx_data.append(fx_df)

    if not all_fx_data:
        return pd.DataFrame(columns=["timestamp", "close", "currency"])
    return pd.concat(all_fx_data, ignore_index=True)

def convert_ordinary_closes_to_usd(
    raw_ordinary_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    adr_info_path: str | Path | None = None,
    fx_dir: str | Path | None = None,
    share_reclass_path: str | Path | None = None,
    splits_path: str | Path | None = None,
    adjustments: tuple[str, ...] = ("none", "all"),
    repo_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_ordinary_dir) if raw_ordinary_dir is not None else resolve_repo_path(
        "data", "raw", "ordinary", repo_root=repo_root
    )
    output_base = Path(output_dir) if output_dir is not None else resolve_repo_path(
        "data", "processed", "ordinary", repo_root=repo_root
    )
    fx_base = Path(fx_dir) if fx_dir is not None else resolve_repo_path(
        "data", "raw", "currencies", "minute_bars", repo_root=repo_root
    )

    adr_info = load_adr_info(adr_info_path=adr_info_path, repo_root=repo_root)
    exchanges = adr_info["exchange"].dropna().unique().tolist()
    outputs: dict[str, pd.DataFrame] = {}

    for adjust in adjustments:
        raw_local_price_file = raw_dir / f"ord_PX_LAST_adjust_{adjust}.csv"
        output_file = output_base / f"ord_close_to_usd_adr_PX_LAST_adjust_{adjust}.csv"

        local_price_df = pd.read_csv(raw_local_price_file, index_col=0, parse_dates=True).sort_index()
        start_date = local_price_df.index[0].strftime("%Y-%m-%d")
        end_date = local_price_df.index[-1].strftime("%Y-%m-%d")

        close_time = pd.DataFrame(
            {
                exchange: mcal.get_calendar(exchange)
                .schedule(start_date=start_date, end_date=end_date)["market_close"]
                .dt.tz_convert("America/New_York")
                for exchange in exchanges
            }
        )
        close_time = close_time.stack().reset_index(name="close_time").rename(
            columns={"level_0": "date", "level_1": "exchange"}
        )

        needed_currencies = sorted(
            set(adr_info["currency"].dropna().unique().tolist()).intersection(SUPPORTED_FX_CURRENCIES)
        )
        fx_prices = load_fx_prices(fx_base, needed_currencies)

        stacked_price = (
            local_price_df.stack().reset_index(name="price").rename(columns={"level_0": "date", "level_1": "ticker"})
        )
        sh_per_adr = get_sh_per_adr(
            start_date,
            end_date,
            for_adjusted=(adjust != "none"),
            adr_info_path=adr_info_path,
            share_reclass_path=share_reclass_path,
            splits_path=splits_path,
            repo_root=repo_root,
        )
        sh_per_adr = sh_per_adr.rename(columns=adr_info.set_index("adr_ticker")["id"].to_dict())
        sh_per_adr = sh_per_adr.stack().to_frame(name="sh_per_adr").reset_index(names=["date", "ticker"])

        stacked_price = pd.merge(stacked_price, sh_per_adr, on=["date", "ticker"], how="left")
        stacked_price = pd.merge(
            stacked_price,
            adr_info[["id", "exchange", "currency"]],
            left_on="ticker",
            right_on="id",
            how="left",
        ).drop(columns=["id"])
        stacked_price = pd.merge(stacked_price, close_time, on=["date", "exchange"], how="left")
        stacked_price = stacked_price[stacked_price["currency"].isin(needed_currencies)]

        stacked_price = pd.merge(
            stacked_price,
            fx_prices[["timestamp", "close", "currency"]].rename(columns={"close": "fx_rate"}),
            left_on=["close_time", "currency"],
            right_on=["timestamp", "currency"],
            how="left",
        ).drop(columns=["timestamp"])
        stacked_price["price_usd"] = stacked_price["price"] * stacked_price["fx_rate"]
        stacked_price["adr_equivalent_price_usd"] = stacked_price["price_usd"] * stacked_price["sh_per_adr"]

        price_df = stacked_price.pivot(index="date", columns="ticker", values="adr_equivalent_price_usd")
        ensure_parent_dir(output_file)
        price_df.to_csv(output_file)
        outputs[adjust] = price_df
        print(f"Converted ordinary close prices ({adjust}) to {output_file}")

    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert ordinary close prices to ADR-equivalent USD prices.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--raw-ordinary-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adr-info", default=None)
    parser.add_argument("--fx-dir", default=None)
    parser.add_argument("--share-reclass", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--adjustments", nargs="+", default=None)
    args = parser.parse_args(argv)

    convert_ordinary_closes_to_usd(
        raw_ordinary_dir=args.raw_ordinary_dir,
        output_dir=args.output_dir,
        adr_info_path=args.adr_info,
        fx_dir=args.fx_dir,
        share_reclass_path=args.share_reclass,
        splits_path=args.splits,
        adjustments=tuple(args.adjustments) if args.adjustments is not None else ("none", "all"),
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
