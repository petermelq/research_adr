This repository contains a DVC pipeline that processes stock market and futures data, and simulates ADR arbitrage strategies using the repo /home/pmalonis/backtester.

## Strategy Overview

The strategy exploits intra-day price convergence between ADRs (US-traded) and their underlying ordinary stocks (foreign-traded). When a foreign market closes, the ordinary stock closing price is combined with futures index movements and a beta model to predict where the ADR should be priced. Trades are entered at 14:30 UTC if the signal is strong enough, and exited at US market close. Portfolio construction uses CVXPY convex optimization.

The ordinary stock is the stock traded in foreign markets. The ADRs are corresponding stocks traded in US markets.

## Backtester Integration

Backtester (/home/pmalonis/backtester) is editably pip installed and provides the engine for running backtests. The specific strategy logic and data loading is accomplished by extending backtester with custom classes (strategy classes in `src/backtest_strategies/` and dataloader classes). Backtester only needs to be updated as a last resort. A guide for creating custom strategies can be found in /home/pmalonis/backtester/CUSTOM_STRATEGY_GUIDE.md.

**Important coupling**: `generate_trades()` parameter names in strategy classes must match data loader keys in the YAML config exactly. The backtester validates this at startup.

## Active Configs and Strategies

- `hedged_single_time.yaml` and `db_hedged_single_time.yaml` are the main active backtest configs.
- Recent backtest configs (with parameters used) are saved in `adr_results/` subdirectories.
- `var_penalty` and `p_volume` (position size as fraction of ADV) are actively being tuned.

## Two Signal Paths

- **fixed_time_signal** (main path): Uses First Rate Data futures — long history, continuous series with price adjustment. Issue: FRD active contract selection isn't always the most liquid contract; European futures after Europe close can have thin liquidity.
- **db_fixed_time_signal**: Uses Databento futures — raw per-contract data, can select by liquidity. Only goes back to March 2025 for Eurex. No price adjustment needed since only intraday returns are used.

## Signal Development Direction

Currently working on incorporating US stocks as predictors in addition to international futures. May also try incorporating trading volume, order flow imbalance, and international ETF premia/discounts (particularly the hedging ETFs).

## Performance Targets

- Sharpe ratio above 3
- Return on GMV of 5% or greater (annualized average of daily PnL / max GMV for that day)
- `notebooks/bt_with_fees2.ipynb` is the most current evaluation notebook.

## Date Ranges

Two date ranges coexist: `params.yaml` has wide dates (2018-2026) for pipeline data processing to support lookback windows, while backtest configs have narrower ranges (e.g. 2023-2025) for evaluation.

## Data Sources and Key Files

- `data/raw/` contains various CSVs created by `notebooks/get_adr_info.ipynb` — ADR info including US tickers, ordinary (foreign) tickers, exchanges.
- `data/raw/close_time_offsets.csv` contains exchange-specific offsets (compatible with pandas Timedelta) for closing auction timing. These are added to closing times from `pandas_market_calendars`.
- `share_reclass.csv`: Handles ADR ratio changes. Uncommon, requires manual data entry.
- **Databento**: ADR TCBBO, 1-min BBO (multi-exchange), futures BBO/OHLCV.
- **Bloomberg (linux_xbbg)**: Daily closing prices for foreign ordinary shares, corporate actions. Fragile but don't modify.
- **First Rate Data**: Russell 1000 daily OHLCV, futures continuous contracts.

## Market Calendar and Closing Times

The repo uses `pandas_market_calendars` for market hours and holidays. The closing times from pandas_market_calendars aren't exact because there is an offset between the exchange closing time and when the closing auction takes place. There is also randomness in when the closing auction happens for certain exchanges, so the closest minute at or after the auction time is used. `data/raw/close_time_offsets.csv` encodes these offsets per exchange.

## Development Workflow

- `dvc repro` detects which stages need rerun. For signal changes: any new data stages + signal generation + backtest.
- Signal generation and backtest strategy/evaluation are the most actively changing parts. Data download, NBBO consolidation, and currency conversion are stable.
- Hard-coded date exclusions are ad-hoc — leave as-is.
- When making changes: write a detailed plan, suggest unit tests, then ask for feedback before implementing.