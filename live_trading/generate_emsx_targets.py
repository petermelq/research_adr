#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

DEFAULT_ADR_INFO = DATA_ROOT / "raw" / "adr_info.csv"
DEFAULT_ADR_TRADE_PRICE = DATA_ROOT / "processed" / "adrs" / "adr_daily_fixed_time_mid.csv"
DEFAULT_ADR_CLOSE = DATA_ROOT / "raw" / "adrs" / "adr_PX_LAST_adjust_none.csv"
DEFAULT_TURNOVER = DATA_ROOT / "raw" / "adrs" / "adr_turnover.csv"
DEFAULT_ETF_TRADE_PRICE = DATA_ROOT / "processed" / "etfs" / "market" / "market_etf_daily_fixed_time_mid.csv"
DEFAULT_ETF_CLOSE = DATA_ROOT / "raw" / "etfs" / "market" / "market_etf_PX_LAST_adjust_none.csv"
DEFAULT_HEDGE_RATIOS = DATA_ROOT / "processed" / "market_etf_hedge_ratios.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "live_trading" / "output"

DATE_EXCLUSION_START = pd.Timestamp("2025-04-04")
DATE_EXCLUSION_END = pd.Timestamp("2025-04-08")
BP_SHEL_EXCLUSION_DATE = pd.Timestamp("2025-06-25")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EMSX-style ADR target orders from Redis alpha snapshots.",
    )
    parser.add_argument("--trading-day", type=str, default=None, help="Trading day in YYYY-MM-DD. Defaults to latest common date in local inputs.")
    parser.add_argument("--alpha-key-template", type=str, required=True, help="Redis key template, for example 'adr:alphas:{date}'.")
    parser.add_argument("--date-format", type=str, default="%Y-%m-%d", help="Date format used when expanding {date} in the Redis key template.")
    parser.add_argument("--redis-url", type=str, default=os.environ.get("REDIS_URL"), help="Redis URL. Defaults to REDIS_URL.")
    parser.add_argument("--redis-host", type=str, default=os.environ.get("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.environ.get("REDIS_PORT", "6379")))
    parser.add_argument("--redis-db", type=int, default=int(os.environ.get("REDIS_DB", "0")))
    parser.add_argument("--redis-password", type=str, default=os.environ.get("REDIS_PASSWORD"))
    parser.add_argument("--alpha-format", choices=["auto", "hash", "json"], default="auto", help="Redis payload format per key.")
    parser.add_argument("--var-penalty", type=float, default=0.0001)
    parser.add_argument("--p-volume", type=float, default=0.02)
    parser.add_argument("--vol-lookback", type=int, default=100)
    parser.add_argument("--turnover-lookback", type=int, default=90)
    parser.add_argument("--position-limit", type=float, default=1e6)
    parser.add_argument("--hedged", action="store_true", help="Include market ETF hedge orders.")
    parser.add_argument("--limit-cost", type=float, default=0.0, help="Signed limit adjustment versus trade price.")
    parser.add_argument("--order-type", type=str, default="MOC")
    parser.add_argument("--tif", type=str, default="DAY")
    parser.add_argument("--broker", type=str, default="")
    parser.add_argument("--account", type=str, default="")
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--current-positions", type=Path, default=None, help="Optional CSV with columns ticker,position or Symbol,FIRM_POS.")
    parser.add_argument("--adr-info", type=Path, default=DEFAULT_ADR_INFO)
    parser.add_argument("--adr-trade-price", type=Path, default=DEFAULT_ADR_TRADE_PRICE)
    parser.add_argument("--adr-close", type=Path, default=DEFAULT_ADR_CLOSE)
    parser.add_argument("--turnover", type=Path, default=DEFAULT_TURNOVER)
    parser.add_argument("--etf-trade-price", type=Path, default=DEFAULT_ETF_TRADE_PRICE)
    parser.add_argument("--etf-close", type=Path, default=DEFAULT_ETF_CLOSE)
    parser.add_argument("--hedge-ratios", type=Path, default=DEFAULT_HEDGE_RATIOS)
    return parser.parse_args()


def read_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    df.columns = [str(c).strip() for c in df.columns]
    return df.sort_index()


def build_schedule(adr_info: pd.DataFrame) -> pd.DataFrame:
    schedules = []
    for exchange in list(adr_info["exchange"].unique()) + ["XNYS"]:
        cal = mcal.get_calendar(exchange)
        sched = cal.schedule("1980-01-01", "2030-01-01").rename(columns={"market_close": exchange})
        sched = sched[[exchange]]
        schedules.append(sched)

    schedule = pd.concat(schedules, axis=1)
    for exchange in schedule.columns:
        null_mask = schedule[exchange].isnull()
        schedule.loc[null_mask, exchange] = schedule.loc[null_mask].index.tz_localize("UTC")
        cal = mcal.get_calendar(exchange)
        schedule[exchange] = schedule[exchange].dt.tz_convert(cal.tz).dt.time
    return schedule


def latest_common_date(frames: list[pd.DataFrame]) -> pd.Timestamp:
    common_dates = set(frames[0].index)
    for frame in frames[1:]:
        common_dates &= set(frame.index)
    if not common_dates:
        raise ValueError("No common dates across required input files.")
    return max(common_dates)


def normal_close_tickers(adr_info: pd.DataFrame, schedule: pd.DataFrame, trading_day: pd.Timestamp) -> list[str]:
    merged = pd.merge(
        adr_info[["adr", "normal_close_time", "exchange"]],
        schedule.loc[trading_day].rename("day_close_time"),
        left_on="exchange",
        right_index=True,
    )
    tickers = merged.loc[merged["day_close_time"] == merged["normal_close_time"], "adr"]
    return tickers.str.replace(" US Equity", "", regex=False).tolist()


def ny_normal_close(schedule: pd.DataFrame, trading_day: pd.Timestamp) -> bool:
    return schedule.loc[trading_day, "XNYS"] == mcal.get_calendar("XNYS").close_time


def redis_client_from_args(args: argparse.Namespace):
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("The 'redis' package is required to load alphas from Redis.") from exc

    if args.redis_url:
        return redis.Redis.from_url(args.redis_url, decode_responses=True)
    return redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password,
        decode_responses=True,
    )


def _coerce_alpha_mapping(payload: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        ticker = str(key).replace(" US Equity", "").strip()
        if value in (None, ""):
            continue
        try:
            out[ticker] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def load_alpha_for_date(redis_client, key: str, alpha_format: str) -> dict[str, float]:
    if alpha_format in ("auto", "hash"):
        raw_hash = redis_client.hgetall(key)
        if raw_hash:
            return _coerce_alpha_mapping(raw_hash)
        if alpha_format == "hash":
            return {}

    raw_value = redis_client.get(key)
    if raw_value is None:
        return {}
    if alpha_format not in ("auto", "json"):
        return {}

    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in Redis key {key}, got {type(parsed).__name__}.")
    return _coerce_alpha_mapping(parsed)


def load_alpha_history(
    redis_client,
    key_template: str,
    dates: list[pd.Timestamp],
    date_format: str,
    alpha_format: str,
) -> pd.DataFrame:
    rows = {}
    for trading_day in dates:
        key = key_template.format(date=trading_day.strftime(date_format))
        rows[trading_day] = load_alpha_for_date(redis_client, key=key, alpha_format=alpha_format)
    alpha_df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    alpha_df.index.name = "date"
    return alpha_df


def load_positions(path: Path | None) -> pd.Series:
    if path is None:
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    column_map = {c.lower(): c for c in df.columns}
    ticker_col = column_map.get("ticker") or column_map.get("symbol")
    pos_col = column_map.get("position") or column_map.get("firm_pos") or column_map.get("current_pos")
    if ticker_col is None or pos_col is None:
        raise ValueError("Current positions CSV must include ticker/symbol and position columns.")

    tickers = df[ticker_col].astype(str).str.replace(" US Equity", "", regex=False).str.strip()
    return pd.Series(df[pos_col].astype(float).to_numpy(), index=tickers).groupby(level=0).sum()


def optimize_targets(
    trading_day: pd.Timestamp,
    adr_signal: pd.DataFrame,
    adr_trade_price: pd.DataFrame,
    adr_close: pd.DataFrame,
    turnover_df: pd.DataFrame,
    hedge_ratios: pd.DataFrame,
    etf_trade_price: pd.DataFrame,
    etf_close: pd.DataFrame,
    hedge_dict: dict[str, str],
    var_penalty: float,
    p_volume: float,
    vol_lookback: int,
    turnover_lookback: int,
    position_limit: float,
) -> tuple[pd.DataFrame, pd.Series]:
    merged_prices = pd.merge(
        adr_trade_price.iloc[-vol_lookback - 1 : -1].stack().rename("trade_price"),
        adr_close.iloc[-vol_lookback - 1 : -1].stack().rename("close"),
        left_index=True,
        right_index=True,
    )
    merged_etf_prices = pd.merge(
        etf_trade_price.iloc[-vol_lookback - 1 : -1].stack().rename("trade_price"),
        etf_close.iloc[-vol_lookback - 1 : -1].stack().rename("close"),
        left_index=True,
        right_index=True,
    )

    adr_ret = ((merged_prices["close"] - merged_prices["trade_price"]) / merged_prices["close"]).rename("adr_ret")
    etf_ret = ((merged_etf_prices["close"] - merged_etf_prices["trade_price"]) / merged_etf_prices["close"]).rename("etf_ret")
    hr_stacked = hedge_ratios.iloc[-vol_lookback - 1 : -1].stack().rename("hedge_ratio")

    merged = pd.merge(hr_stacked, adr_ret, left_index=True, right_index=True).reset_index(names=["date", "ticker"])
    merged["hedge_ticker"] = merged["ticker"].map(hedge_dict)
    merged = merged.merge(
        etf_ret.to_frame().reset_index(names=["date", "hedge_ticker"]),
        on=["date", "hedge_ticker"],
        how="inner",
    )
    merged["hedged_ret"] = merged["adr_ret"] - merged["hedge_ratio"] * merged["etf_ret"]
    ret = merged.pivot(index="date", columns="ticker", values="hedged_ret")

    aligned_signal = adr_signal.reindex(index=ret.index, columns=ret.columns)
    res = ret - aligned_signal
    if BP_SHEL_EXCLUSION_DATE in res.index:
        for ticker in ["BP", "SHEL"]:
            if ticker in res.columns:
                res.loc[BP_SHEL_EXCLUSION_DATE, ticker] = 0.0

    exclusion_mask = (res.index >= DATE_EXCLUSION_START) & (res.index <= DATE_EXCLUSION_END)
    res = res.loc[~exclusion_mask]

    tickers = adr_signal.columns.tolist()
    res = res.reindex(columns=tickers)
    cov = cp.psd_wrap(res.fillna(0).cov().values)
    alpha = adr_signal.loc[trading_day].clip(lower=-0.01, upper=0.01).fillna(0).to_numpy()
    turnover = turnover_df.loc[:trading_day, tickers].iloc[-turnover_lookback:-1].mean().fillna(1).to_numpy()

    w = cp.Variable(len(tickers))
    objective = cp.Maximize((alpha @ w) - var_penalty * cp.quad_form(w, cov))
    constraints = [
        cp.multiply(cp.abs(w), 1 / turnover) <= p_volume,
        cp.abs(w) <= position_limit,
    ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", max_iter=100000)
    except Exception:
        problem.solve()

    if w.value is None:
        raise RuntimeError("Optimization failed to produce a solution.")

    weights = pd.DataFrame({"target_notional": w.value}, index=tickers)
    weights["target_notional"] = weights["target_notional"].clip(lower=-2e6, upper=2e6)
    weights["trade_price"] = adr_trade_price.loc[trading_day, tickers]
    weights["hedge_ratio"] = hedge_ratios.loc[trading_day, tickers]
    weights["hedge_ticker"] = weights.index.map(hedge_dict)
    weights["target_shares"] = (weights["target_notional"] / weights["trade_price"]).round()

    weights["hedge_notional"] = weights["target_notional"] * weights["hedge_ratio"] * (-1)
    etf_target_notional = weights.groupby("hedge_ticker")["hedge_notional"].sum().dropna()
    etf_target_shares = (etf_target_notional / etf_trade_price.loc[trading_day, etf_target_notional.index]).round()

    return weights, etf_target_shares


def build_order_rows(target_shares: pd.Series, current_positions: pd.Series, prices: pd.Series) -> pd.DataFrame:
    current_positions = current_positions.reindex(target_shares.index).fillna(0.0)
    rows = []
    for ticker, target_qty in target_shares.items():
        current_qty = float(current_positions.loc[ticker])
        trade_qty = int(round(float(target_qty) - current_qty))
        if trade_qty == 0:
            continue

        if trade_qty > 0:
            rows.append(
                {
                    "Symbol": f"{ticker} US Equity",
                    "Side": "B",
                    "Amount": trade_qty,
                    "reference_price": float(prices.loc[ticker]),
                }
            )
            continue

        sell_qty = abs(trade_qty)
        sell_long = int(min(max(current_qty, 0.0), sell_qty))
        sell_short = int(sell_qty - sell_long)
        if sell_long > 0:
            rows.append(
                {
                    "Symbol": f"{ticker} US Equity",
                    "Side": "S",
                    "Amount": sell_long,
                    "reference_price": float(prices.loc[ticker]),
                }
            )
        if sell_short > 0:
            rows.append(
                {
                    "Symbol": f"{ticker} US Equity",
                    "Side": "SS",
                    "Amount": sell_short,
                    "reference_price": float(prices.loc[ticker]),
                }
            )

    return pd.DataFrame(rows, columns=["Symbol", "Side", "Amount", "reference_price"])


def main() -> None:
    args = parse_args()

    adr_info = pd.read_csv(args.adr_info)
    adr_info["adr_ticker"] = adr_info["adr"].str.replace(" US Equity", "", regex=False)
    adr_info["normal_close_time"] = adr_info["exchange"].apply(lambda exchange: mcal.get_calendar(exchange).close_time)
    hedge_dict = adr_info.set_index("adr_ticker")["market_etf_hedge"].to_dict()
    schedule = build_schedule(adr_info)

    adr_trade_price = read_daily_csv(args.adr_trade_price)
    adr_close = read_daily_csv(args.adr_close)
    turnover_df = read_daily_csv(args.turnover)
    hedge_ratios = read_daily_csv(args.hedge_ratios)
    etf_trade_price = read_daily_csv(args.etf_trade_price)
    etf_close = read_daily_csv(args.etf_close)

    required_frames = [adr_trade_price, adr_close, turnover_df, hedge_ratios, etf_trade_price, etf_close]

    trading_day = pd.Timestamp(args.trading_day).normalize() if args.trading_day else latest_common_date(required_frames)

    if trading_day not in schedule.index:
        raise ValueError(f"{trading_day.date()} is not available in the exchange schedule.")
    if not ny_normal_close(schedule, trading_day):
        raise ValueError(f"{trading_day.date()} is not a normal NYSE close day.")

    trading_tickers = normal_close_tickers(adr_info, schedule, trading_day)
    if len(trading_tickers) < adr_trade_price.shape[1]:
        raise ValueError(f"{trading_day.date()} is not a normal-close day for the full ADR universe.")

    history_dates = [d for d in adr_trade_price.loc[:trading_day].index.tolist()][-args.vol_lookback - 1 :]
    redis_client = redis_client_from_args(args)
    alpha_history = load_alpha_history(
        redis_client=redis_client,
        key_template=args.alpha_key_template,
        dates=history_dates,
        date_format=args.date_format,
        alpha_format=args.alpha_format,
    )
    if trading_day not in alpha_history.index or alpha_history.loc[trading_day].dropna().empty:
        key = args.alpha_key_template.format(date=trading_day.strftime(args.date_format))
        raise ValueError(f"No alpha snapshot found in Redis for {trading_day.date()} at key {key}.")

    cols = adr_trade_price.loc[trading_day].dropna().index.intersection(alpha_history.columns)
    adr_signal = alpha_history.reindex(columns=cols).dropna(how="all", axis=1)

    weights, etf_target_shares = optimize_targets(
        trading_day=trading_day,
        adr_signal=adr_signal,
        adr_trade_price=adr_trade_price.reindex(columns=adr_signal.columns),
        adr_close=adr_close.reindex(columns=adr_signal.columns),
        turnover_df=turnover_df.reindex(columns=adr_signal.columns),
        hedge_ratios=hedge_ratios.reindex(columns=adr_signal.columns),
        etf_trade_price=etf_trade_price,
        etf_close=etf_close,
        hedge_dict=hedge_dict,
        var_penalty=args.var_penalty,
        p_volume=args.p_volume,
        vol_lookback=args.vol_lookback,
        turnover_lookback=args.turnover_lookback,
        position_limit=args.position_limit,
    )

    current_positions = load_positions(args.current_positions)
    adr_target_shares = weights["target_shares"].fillna(0).round().astype(int)
    adr_prices = weights["trade_price"]

    target_rows = pd.DataFrame(
        {
            "ticker": adr_target_shares.index,
            "asset_type": "ADR",
            "target_shares": adr_target_shares.values,
            "target_notional": weights["target_notional"].values,
            "trade_price": adr_prices.values,
            "hedge_ticker": weights["hedge_ticker"].values,
            "hedge_ratio": weights["hedge_ratio"].values,
        }
    )
    if args.hedged and not etf_target_shares.empty:
        etf_targets = pd.DataFrame(
            {
                "ticker": etf_target_shares.index,
                "asset_type": "ETF",
                "target_shares": etf_target_shares.astype(int).values,
                "target_notional": (
                    etf_target_shares.astype(float).to_numpy()
                    * etf_trade_price.loc[trading_day, etf_target_shares.index].to_numpy()
                ),
                "trade_price": etf_trade_price.loc[trading_day, etf_target_shares.index].to_numpy(),
                "hedge_ticker": "",
                "hedge_ratio": np.nan,
            }
        )
        target_rows = pd.concat([target_rows, etf_targets], ignore_index=True)

    order_frames = [build_order_rows(adr_target_shares, current_positions, adr_prices)]
    if args.hedged:
        etf_prices = etf_trade_price.loc[trading_day, etf_target_shares.index]
        order_frames.append(build_order_rows(etf_target_shares.astype(int), current_positions, etf_prices))

    orders = pd.concat(order_frames, ignore_index=True)
    if not orders.empty:
        sign = orders["Side"].map({"B": 1, "S": -1, "SS": -1})
        orders["Order Type"] = args.order_type
        orders["Broker"] = args.broker
        orders["TIF"] = args.tif
        orders["Account"] = args.account
        orders["Limit"] = (orders["reference_price"] * (1 + sign * args.limit_cost)).round(2)
        orders = orders[["Symbol", "Order Type", "Side", "Amount", "Limit", "TIF", "Broker", "Account"]]
    else:
        orders = pd.DataFrame(columns=["Symbol", "Order Type", "Side", "Amount", "Limit", "TIF", "Broker", "Account"])

    output_dir = args.output_dir / trading_day.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"{args.output_prefix}targets_{trading_day:%Y-%m-%d}.csv"
    orders_path = output_dir / f"{args.output_prefix}emsx_order_{trading_day:%Y-%m-%d}.csv"

    target_rows.to_csv(target_path, index=False)
    orders.to_csv(orders_path, index=False)

    print(f"Saved targets to {target_path}")
    print(f"Saved EMSX orders to {orders_path}")


if __name__ == "__main__":
    main()
