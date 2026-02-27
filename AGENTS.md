# AGENTS.md — ADR Arbitrage Pipeline Reference

This document enables agents to work on this codebase with minimal additional context. Read CLAUDE.md first for project goals and workflow rules, then use this as the technical reference.

## Defaults

- **Inference**: Use full minute inference by default. Only run 30-minute inference (`--eval-times-only`) when explicitly requested.
- **Evaluation**: Report mean per-ticker IC as the default model comparison metric. Include pooled IC only as secondary/context unless explicitly requested.

## Architecture Overview

A DVC pipeline that: downloads market data → processes/aligns it → generates trading signals → runs backtests. The signal predicts ADR mispricing relative to the underlying ordinary stock using futures index movements and (optionally) US stock cross-predictors.

```
Data Downloads (Bloomberg, Databento, FRD)
    ↓
Price Alignment & Currency Conversion
    ↓
Beta Models (index-only, Russell rolling betas)
    ↓
Signal Generation (futures-only baseline, or model-augmented)
    ↓
Fixed-Time Extraction (minute signal → daily signal at 14:00 ET)
    ↓
Backtest (backtester_run with YAML config)
```

## Signal Mathematics

### Core equation (futures-only baseline)
```
signal(t) = futures_return(close→t) × beta - adr_return(close→t)
```
Where `close` is the foreign exchange closing auction time, `t` is the current US minute, `futures_return` is the USD-converted index futures return, and `adr_return` is the ADR mid price return. A positive signal means the ADR has lagged the predicted move and should mean-revert upward.

### Russell-augmented signal
```
russell_residual(i) = russell_return(i) - beta(i) × index_return
augmented_signal = baseline_signal + model(russell_residuals)
```
The model (PCR, Ridge, etc.) predicts the ordinary stock's idiosyncratic return from ~996 Russell 1000 stock residuals. This prediction gets added to the futures-only baseline.

### Residualization
All returns are residualized against the relevant index to isolate alpha. Currency conversion is applied when stock and index are in different currencies:
```
r_usd = (1 + r_native) × (1 + r_fx) - 1
```

## DVC Pipeline Stages (Grouped by Function)

### Layer 1: Data Downloads (stable, rarely changed)

**Bloomberg downloads** (via `data_tools.cli.download_bbg_daily` using `linux_xbbg`):
- `download_adr_close` / `download_adr_open` / `adj_adr_close` — ADR daily prices (PX_LAST, PX_OPEN; unadjusted and split+div adjusted)
- `download_ord_close` / `download_ord_open` / `adj_download_ord_close` / `split_adj_download_ord_close` — Ordinary stock daily prices
- `download_market_etf_close` / `download_market_etf_open` / `adj_market_etf_close` — Market hedge ETF daily prices
- `download_sector_etf_close` — Sector ETF daily prices
- `download_adr_turnover` / `download_russell1000_turnover` — Trading volume data
- `download_index_close` — Cash index levels (UKX, SX5E, NKY, etc.)
- `download_close_times` — Exchange close times from Bloomberg
- `earnings` — Earnings announcement dates

All Bloomberg stages read `data/raw/adr_info.csv` for tickers and use `--tickers_columns` to select which column. The `--field` flag selects the Bloomberg field. `--adjust=all` applies split+dividend adjustments; `--adjust=split` applies split-only; default is unadjusted.

**WARNING**: `linux_xbbg` is fragile. It shells out to Windows to run Bloomberg API calls. Do not modify its invocations or the library itself.

**Databento downloads** (via `data_tools.cli.batch_download_to_parquet_with_append`):
- `download_adr_tcbbo` — ADR trade+BBO tick data (XNAS.BASIC dataset)
- `download_adr_bbo-1m` / `download_market_etf_bbo-1m` / `download_sector_etf_bbo-1m` — 1-min BBO from XNYS.PILLAR, XNAS.ITCH, ARCX.PILLAR
- `download_eurex_bbo-1m` / `download_ice_bbo-1m` / `download_cme_bbo-1m` — Futures BBO by exchange
- `download_eurex_ohlcv-1m` / `download_ice_ohlcv-1m` / `download_cme_ohlcv-1m` — Futures OHLCV by exchange

All Databento stages use `--partition_period=none` for minute data (one file per ticker) and output to `data/raw/`. `persist: true` means DVC won't delete these outputs.

**First Rate Data downloads**:
- `download_russell1000_frd_ohlcv-1m` — Russell 1000 constituent minute bars via `data_tools.cli.frd_download_to_parquet_with_append`

Outputs to `data/raw/russell1000/ohlcv-1m/ticker={TICKER}/data.parquet`. Format: DatetimeIndex (naive ET), columns: Open, High, Low, Close, Volume, date.

**FX minute bars** (from histdata.com):
- `download_scandi_usd_fx_bars` — Downloads USDNOK, USDSEK, USDDKK
- `usdjpy_to_jpyusd` / `usdchf_to_chfusd` / `invert_scandi_usd_fx_bars` — Inverts FX pairs to get xxxUSD format

FX files are at `data/raw/currencies/minute_bars/{PAIR}_full_1min.txt`. Format: CSV without header, columns: date, time, open, high, low, close, volume.

### Layer 2: NBBO Consolidation (stable)

- `create_nbbo_adr_bbo-1m` / `create_nbbo_market_etf_bbo-1m` / `create_nbbo_sector_etf_bbo-1m` — Consolidates per-exchange BBO into NBBO (best bid/ask across exchanges). Uses `data_tools.cli.nbbo_consolidate_bbo_1m_with_append`. Output: `data/raw/{adrs,etfs}/bbo-1m/nbbo/ticker={TICKER}/data.parquet`.

### Layer 3: Price Processing & Alignment (stable)

**Corporate actions**:
- `stock_splits` / `stock_dividends` — Downloads splits/dividends via Bloomberg for ADRs and sector ETFs
- `adr_adjustments` / `market_etf_adjustments` / `sector_etf_adjustments` — Computes cumulative adjustment factors from unadjusted price ratios

**Currency conversion** (ordinary → USD):
- `convert_ord_currency_close` / `convert_ord_currency_open` — Converts ordinary stock prices from local currency to USD at the exchange close/open time using FX minute bars. Applies ADR ratio (`sh_per_adr`). Output: `data/processed/ordinary/ord_close_to_usd_adr_PX_LAST_adjust_*.csv`.

**Russell 1000 processing**:
- `build_russell1000_tickers` — Extracts unique tickers from `data/raw/historical_russell_1000.csv`
- `russell1000_close_at_exchange_auction` — For each foreign exchange, samples Russell minute bars at that exchange's closing auction time. Output: `data/processed/russell1000/close_at_exchange_auction/{EXCHANGE}.csv` (index=date, columns=tickers)
- `russell1000_adjustments` / `russell1000_close_at_exchange_auction_adjusted` — Applies corporate action adjustments to Russell close prices

**ADR mid prices**:
- `adr_mid_at_ordinary_auction` — ADR NBBO mid price sampled at each foreign exchange's closing auction time. Output: `data/processed/adrs/adr_mid_at_ord_auction_adjust_*.csv`
- `adr_daily_mid` — ADR NBBO mid at fixed trade time (14:00 ET). Output: `data/processed/adrs/adr_daily_fixed_time_mid.csv`
- `adr_open_mid` — ADR NBBO mid at 9:31 ET
- `adr_mid_ny_open` — ADR NBBO mid at 9:35 ET
- `daily_adr_for_cov` — ADR mid at 13:00 ET (for covariance estimation)
- `market_etf_daily_fixed_time_mid` — Market ETF mid at fixed trade time

**Index price alignment**:
- `build_aligned_index_prices` — For stocks whose exchange close doesn't align with the index close, samples futures price at the exchange close time as a proxy. Output: `data/processed/aligned_index_prices.csv`

**Futures conversion to USD**:
- `usd_index_futures` — FRD continuous futures → USD using FX rates and notional multipliers (FTUK=10, FDAX=25, FCE=10, FXXP=50, FESX=10, FTI=200, NIY=500). Output: `data/processed/futures/converted_minute_bars/symbol={SYM}/*.parquet`
- `bbo_usd_index_futures` — Same for Databento BBO futures (Z, FESX, NIY only). Output: `data/processed/futures/converted_bbo/`
- `ohlcv-1m_usd_index_futures` — Same for Databento OHLCV futures
- `active_futures_series` / `ohlcv-1m_active_futures_series` — Selects most liquid contract per date from raw Databento data using daily volume
- `daily_futures_volume` — Downloads daily volume per futures contract via Bloomberg

**Other**:
- `market_etf_hedge_ratios` — Rolling beta of each ADR vs its market ETF from adjusted daily close prices. Output: `data/processed/market_etf_hedge_ratios.csv`
- `adr_open_mean_premium` — Rolling mean ADR premium at open
- `minute_volume_stats` — Per-minute average dollar volume from TCBBO data

### Layer 4: Beta Models (moderately stable)

- `index_only_model` (`get_betas_no_sector.py`) — Rolling 280-day (configurable via `pred.lookback_days`) beta of each ordinary stock vs its index. Uses aligned index prices and currency-adjusted returns. Output: `data/processed/models/ordinary_betas_index_only.csv`. This is the **core risk model** used in futures-only signal generation.

- `compute_russell_betas` (`compute_russell_betas.py`) — Rolling 250-day beta of each Russell 1000 stock vs its exchange's index. Handles Asia differently (uses daily close-to-close vs futures). Output: `data/processed/russell1000/russell_betas/{EXCHANGE}.parquet`. Used to residualize Russell returns for model features.

### Layer 5: Signal Generation (actively changing)

**Futures-only baseline**:
- `only_futures_full_signal` (`only_futures_full_signal.py`) — Generates minute-level signal for each ADR ticker. Uses FRD futures. Output: `data/processed/futures_only_signal/ticker={TICKER}/data.parquet`
- `db_only_futures_full_signal` — Same using Databento futures. Output: `data/processed/db_futures_only_signal/`

**Russell-augmented models** (PCR is primary, others are experimental comparisons):
- `prepare_russell_features_extended` — Builds per-ADR-ticker feature DataFrames: target = ordinary residual, features = ~996 Russell stock residuals (at exchange close). Applies coverage-based filtering (drops features with <30% total coverage or <50% recent coverage). Two paths: "local" for non-Asia ADRs (ordinary close-to-close returns) and "NY close" for Asia ADRs (NY close-to-close returns). Supports `--ny-close-target-tickers` to override which ADRs use the NY path. Output: `data/processed/models/with_us_stocks/features_extended/{TICKER}.parquet`
- `{model}_train_models` — Walk-forward train/val/test. Currently 20 train months + 6 val months → deploy on following month. PCR fits include an intercept (mean of training target) and a configurable `--pcr-min-component-variance` floor (default 1e-4). Output: `data/processed/models/with_us_stocks/{model}/{TICKER}/{YYYY_MM}.pkl`
- `index_russell_{model}_signal` — Applies trained model to live Russell minute returns + baseline signal. Output: `data/processed/index_russell_{model}_signal/`
- `index_russell_{model}_signal_gated` (`gate_russell_experiment_signal.py`) — For each test window, uses model signal only when val IC > baseline val IC, otherwise falls back to baseline
- `evaluate_{model}_notebook` — Runs IC comparison notebook template

**Model types available**: ridge, pcr, pcr_old_universe, pls, elasticnet, rrr, robust_pcr, huber, random_forest. **PCR (pcr_old_universe specifically) is the actively deployed model.** Others exist for comparison.

**Old universe note**: `pcr_old_universe` uses a fixed Russell 1000 constituent list (`data/raw/old_russell1000_tickers.csv`) for feature selection, matching the original pipeline behavior. The current pipeline dynamically selects constituents based on the validation period end date using `data/raw/historical_russell_1000.csv`. The old universe variant exists for comparison and will eventually be removed.

**Feature quality constants** (in `prepare_russell_features_extended.py`):
| Constant | Value | Meaning |
|----------|-------|---------|
| `FEATURE_RECENT_DAYS` | 60 | Window for recent coverage check |
| `FEATURE_MIN_TOTAL_COVERAGE` | 0.30 | Min non-null fraction over entire period |
| `FEATURE_MIN_RECENT_COVERAGE` | 0.50 | Min non-null fraction in recent window |
| `FEATURE_MIN_RECENT_OBS` | 20 | Min non-null count in recent window |
| `FEATURE_MIN_COUNT` | 20 | Min Russell features per ADR after filtering |
| `FEATURE_MIN_SUCCESS_RATIO` | 0.85 | Min fraction of ADRs that must produce features |

### Layer 6: Fixed-Time Signal Extraction

- `fixed_time_signal` — Extracts futures-only signal at 14:00 ET. Output: `data/processed/fixed_time_signal.csv`
- `pcr_fixed_time_signal` — Extracts PCR old universe gated signal at 14:00 ET. Output: `data/processed/pcr_fixed_time_signal.csv`
- `ridge_fixed_time_signal` / `db_fixed_time_signal` — Same for other signal variants

All use `from_full_futures_fixed_time_signal.py`: reads minute signal parquets, filters to before `fixed_trade_time` (14:00), takes last value per day. Output format: CSV with index=date, columns=ADR tickers.

### Layer 7: Backtests

- `hedged_backtest_fixed_time` — Main active backtest using `hedged_single_time.yaml` + `pcr_fixed_time_signal.csv`
- `db_hedged_backtest_fixed_time` — Databento variant
- `brit_backtest_fixed_time` — Unhedged variant
- `minute_vwap_hedged_ADR_backtest` — VWAP accumulation variant (experimental)

## Key Data Files Reference

| File | Format | Description |
|------|--------|-------------|
| `data/raw/adr_info.csv` | CSV, 20 cols | Master ADR mapping: ticker, exchange, currency, sh_per_adr, index_future_bbg, market_etf_hedge |
| `data/raw/futures_symbols.csv` | CSV | Maps exchange symbols → Bloomberg/FRD symbols, underlying index, currency |
| `data/raw/close_time_offsets.csv` | CSV | Per-exchange offset (e.g., "6min") added to pandas_market_calendars close time |
| `data/raw/historical_russell_1000.csv` | CSV | Historical Russell 1000 membership with dates, weights, positions |
| `data/raw/sector_etfs.csv` | CSV | ADR → sector hedge ETF mapping |
| `data/raw/adrs/share_reclass.csv` | CSV | ADR ratio changes (rare, manual) |
| `params.yaml` | YAML | Global pipeline params (dates, lookback windows, trade time) |

### Exchange → Futures → ETF Mapping

| Exchange | Futures (BBG) | FRD Symbol | Underlying Index | Hedge ETF | Close Offset |
|----------|--------------|------------|-----------------|-----------|-------------|
| XLON | Z | FTUK | UKX | EWU | 6min |
| XAMS, XBRU, XPAR, XETR, XMAD, XMIL (most Europe) | VG | FESX | SX5E | FEZ | 6min (varies) |
| XTKS | NH | NIY | NKY | EWJ | 1min |
| XASX | (none active) | (none) | (none) | EWA | 11min |

**IMPORTANT**: VG maps to SX5E (Euro Stoxx 50), NOT SXXP (STOXX Europe 600). The `index_future_bbg` column in adr_info.csv has trailing spaces for some values (e.g., `'Z '`).

### Signal Data Formats

| Signal | Path Pattern | Index | Columns | dtype |
|--------|-------------|-------|---------|-------|
| Full minute signal | `data/processed/{signal_name}/ticker={TICKER}/data.parquet` | DatetimeIndex (tz-aware ET) | signal, date | float32 or float64 |
| Fixed-time daily signal | `data/processed/{name}_fixed_time_signal.csv` | date | ADR tickers | float64 |

## Backtest Strategy: hedged_single_time_ADR

### Data Loaders Required (YAML keys must match `generate_trades()` params)

| YAML Key | Source File | Description |
|----------|-----------|-------------|
| `adr_trade_price` | `data/processed/adrs/adr_daily_fixed_time_mid.csv` | ADR mid at 14:00 ET |
| `adr_signal` | `data/processed/pcr_fixed_time_signal.csv` | Trading signal (date × ticker) |
| `adr_close` | `data/raw/adrs/adr_PX_LAST_adjust_none.csv` | ADR daily close |
| `turnover_df` | `data/raw/adrs/adr_turnover.csv` | Daily shares traded |
| `etf_trade_price` | `data/processed/etfs/market/market_etf_daily_fixed_time_mid.csv` | ETF mid at 14:00 ET |
| `etf_close` | `data/raw/etfs/market/market_etf_PX_LAST_adjust_none.csv` | ETF daily close |
| `hedge_ratios` | `data/processed/market_etf_hedge_ratios.csv` | Rolling beta vs market ETF |

### Strategy Parameters

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `var_penalty` | 0.0001 | CVXPY variance penalty weight (actively tuned) |
| `p_volume` | 0.02 | Max position as fraction of ADV (actively tuned) |
| `vol_lookback` | 100 | Days for covariance estimation |
| `turnover_lookback` | 90 | Days for average turnover |
| `hedged` | False | Whether to include market ETF hedges (currently off despite filename) |

### Trade Logic Summary
1. At 14:00 ET: read signal, clip to [-0.01, 0.01]
2. Compute covariance from residualized returns (returns minus signal, with hedge adjustment)
3. CVXPY optimization: maximize `signal @ weights - var_penalty * weights^T Cov weights` subject to position size constraints
4. Enter ADR positions (and hedge ETF positions if hedged=True)
5. Exit all positions at 16:00 ET daily close

## Key Source Files

| File | Purpose | Changes Often? |
|------|---------|---------------|
| `src/only_futures_full_signal.py` | Futures-only baseline signal | Rare |
| `src/index_russell_experiment_signal.py` | Model-augmented signal generation | Yes |
| `src/train_russell_experiment_models.py` | Walk-forward model training | Yes |
| `src/prepare_russell_features_extended.py` | Feature matrix construction | Moderate |
| `src/gate_russell_experiment_signal.py` | Model vs baseline switching | Rare |
| `src/compute_russell_betas.py` | Russell stock betas | Rare |
| `src/get_betas_no_sector.py` | Ordinary stock betas (baseline) | Rare |
| `src/utils_lasso_residuals.py` | Shared: residualization, FX, alignment | Moderate |
| `src/utils.py` | Shared: calendar, params, partitions | Rare |
| `src/from_full_futures_fixed_time_signal.py` | Minute → daily signal extraction | Rare |
| `src/backtest_strategies/hedged_single_time_ADR.py` | Main backtest strategy | Yes |
| `src/closing_domestic_prices.py` | Ordinary close → USD conversion | Rare |
| `src/market_etf_hedge_ratios.py` | ADR vs ETF hedge betas | Rare |
| `src/adr_mid_at_ordinary_auction.py` | ADR mid at foreign close | Rare |

## Gotchas and Fragile Areas

- **Bloomberg downloads**: `linux_xbbg` shells out to Windows. Never modify it or its CLI invocations.
- **DVC global cache**: Located at `~/.cache/dvc`. Permission issues can occur when sandboxed agents run `dvc repro`. If `dvc repro` fails with permission errors, try running with `dangerouslyDisableSandbox: true` or check that `~/.cache/dvc` is accessible.
- **`adr_info['adr']`** has ` US Equity` suffix that needs stripping: `adr_info['adr'].str.replace(' US Equity', '')`.
- **`index_future_bbg`** has trailing spaces for some values (e.g., `'Z '`). Strip before matching.
- **Jimmy Carter funeral**: 2025-01-09 must be added as NYSE holiday when using pandas_market_calendars. See `data_tools/src/data_tools/utils.py` for the pattern.
- **Excluded exchanges**: XTKS and XASX close before US extended hours begin, requiring different treatment in Russell feature prep (Asia path in `prepare_russell_features_extended.py` and `compute_russell_betas.py`).
- **Hard-coded date exclusions** in `hedged_single_time_ADR.py`: 2025-04-04 to 2025-04-08, BP/SHEL on 2025-06-25. Leave as-is.
- **Performance**: Avoid per-minute-per-ticker Python loops for Russell data (996 tickers × ~270 min/day). Vectorize with wide DataFrames and matrix ops. The `groupby` with tz-aware DatetimeIndex can be problematic; iterate over unique dates instead.
- **`features_extended_old_universe`** is a dependency of `pcr_old_universe_train_models` but has no DVC stage — it was generated ad-hoc and will be removed in the future.
