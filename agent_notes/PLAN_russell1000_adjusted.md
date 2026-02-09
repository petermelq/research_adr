# Plan: Create Adjusted Russell 1000 Close Prices at Exchange Auction Times

## Overview
Create split/dividend-adjusted versions of Russell 1000 close prices extracted at foreign exchange auction times, following the existing adjustment factor workflow used for ADRs and ETFs.

## Architecture

### 1. Download Adjustment Factors (One-time, DVC Stage)
**Script**: `src/get_russell1000_adjustments.py`
**DVC Stage**: `russell1000_adjustments`

**Approach**:
- Similar to `src/get_adjustments.py` but simplified since Russell 1000 stocks don't have the ADR-specific complications (no ratio changes, no foreign dividends)
- Use `linux_xbbg.blp.bds()` with `EQY_DVD_ADJUST_FACT` and filter `NORMAL_CASH|ABNORMAL_CASH|CAPITAL_CHANGE`
- **One-time execution**: This stage should only run once to avoid hitting Bloomberg data limits
- Use existing unadjusted data from any source to determine date range and tickers

**Implementation Details**:
```python
# Download adjustment factors using linux_xbbg
adj_factors = blp.bds(
    bbg_tickers,
    'EQY_DVD_ADJUST_FACT',
    Corporate_Actions_Filter='NORMAL_CASH|ABNORMAL_CASH|CAPITAL_CHANGE'
)
```

**Input**:
- `data/raw/russell1000_tickers.csv` - List of Russell 1000 tickers
- `params.yaml` - Date range (`frd_start_date`, `end_date`)

**Output**:
- `data/processed/russell1000/adjustment_factors.csv`
  - Columns: ticker, adjustment_date, adjustment_factor, adjustment_factor_operator_type, adjustment_factor_flag
  - This mirrors Bloomberg's EQY_DVD_ADJUST_FACT structure

**DVC Configuration**:
```yaml
russell1000_adjustments:
  cmd: python src/get_russell1000_adjustments.py
  deps:
    - src/get_russell1000_adjustments.py
    - data/raw/russell1000_tickers.csv
  params:
    - frd_start_date
    - end_date
  outs:
    - data/processed/russell1000/adjustment_factors.csv
```

### 2. Apply Adjustments to Create Adjusted Series (DVC Stage)
**Script**: `src/russell1000_close_at_exchange_auction_adjusted.py`
**DVC Stage**: `russell1000_close_at_exchange_auction_adjusted`

**Approach**:
- Read unadjusted prices from `data/processed/russell1000/close_at_exchange_auction/*.csv`
- Read adjustment factors from `data/processed/russell1000/adjustment_factors.csv`
- For each ticker:
  1. Filter adjustment factors for that ticker
  2. Compute daily cumulative adjustment factor using `get_daily_adj()` helper
  3. Multiply prices by cumulative adjustment
- Write adjusted prices to `data/processed/russell1000/close_at_exchange_auction_adjusted/*.csv`

**Key Logic** (following existing pattern):
```python
def get_daily_adj(adj_group, start_date, end_date):
    """
    Convert adjustment factors to daily cumulative adjustment series.

    Logic:
    1. Group by adjustment_date, multiply adjustment factors (handles multiple events same day)
    2. Sort descending (most recent first)
    3. Set adjustment_factor = 1.0 at start_date
    4. Compute cumulative product (creates chain of adjustments)
    5. Shift index back one business day (ex-date -> last cum-dividend date)
    6. Set cum_adj = 1.0 at end_date
    7. Forward-fill daily to create continuous series

    Result: Series where each date has the cumulative adjustment to apply to prices
    """
    adj_df = adj_group.groupby('adjustment_date')[['adjustment_factor']].prod().sort_index(ascending=False)
    adj_df.loc[start_date, 'adjustment_factor'] = 1.0
    adj_df['cum_adj'] = adj_df['adjustment_factor'].cumprod()

    cbday = get_market_business_days('NYSE')
    adj_df.index = [pd.to_datetime(idx) - cbday for idx in adj_df.index]
    adj_df.loc[end_date, 'cum_adj'] = 1.0
    adj_df = adj_df.sort_index().loc[:end_date]
    adj_df = adj_df[['cum_adj']].sort_index().resample('1D').bfill()

    return adj_df
```

**Input**:
- `data/processed/russell1000/close_at_exchange_auction/*.csv` - Unadjusted prices per exchange
- `data/processed/russell1000/adjustment_factors.csv` - Adjustment factors

**Output**:
- `data/processed/russell1000/close_at_exchange_auction_adjusted/*.csv` - One per exchange
  - Same structure as unadjusted: dates as rows, tickers as columns
  - Values are split/dividend-adjusted Close prices

**DVC Configuration**:
```yaml
russell1000_close_at_exchange_auction_adjusted:
  cmd: python src/russell1000_close_at_exchange_auction_adjusted.py
  deps:
    - src/russell1000_close_at_exchange_auction_adjusted.py
    - data/processed/russell1000/close_at_exchange_auction
    - data/processed/russell1000/adjustment_factors.csv
  params:
    - frd_start_date
    - end_date
  outs:
    - data/processed/russell1000/close_at_exchange_auction_adjusted
```

## Unit Test Strategy

**Test File**: `tests/test_russell1000_adjustments.py`

**Test Case**: Validate adjustment logic using existing market ETF data
- **Input**: `data/raw/etfs/market/market_etf_PX_LAST_adjust_none.csv` (unadjusted)
- **Adjustment Factors**: `data/processed/etfs/market/market_etf_adjustment_factors.csv`
- **Expected Output**: `data/raw/etfs/market/market_etf_PX_LAST_adjust_all.csv` (adjusted)

**Test Logic**:
```python
def test_adjustment_factor_application():
    """
    Test that our adjustment logic produces the same results as existing workflow.
    Uses market ETF data since we have both unadjusted and adjusted versions.
    """
    # Read unadjusted prices
    unadj = pd.read_csv('data/raw/etfs/market/market_etf_PX_LAST_adjust_none.csv',
                        index_col=0, parse_dates=True)

    # Read adjustment factors
    adj_factors = pd.read_csv('data/processed/etfs/market/market_etf_adjustment_factors.csv')

    # Apply our adjustment logic
    adj_result = apply_adjustments(unadj, adj_factors, start_date, end_date)

    # Read expected adjusted prices
    expected = pd.read_csv('data/raw/etfs/market/market_etf_PX_LAST_adjust_all.csv',
                          index_col=0, parse_dates=True)

    # Compare (allow small floating point differences)
    pd.testing.assert_frame_equal(adj_result, expected, rtol=1e-6)
```

**Additional Test Cases**:
1. Test with stock split (adjustment_factor = 0.5 for 2:1 split)
2. Test with dividend (adjustment_factor = 1 - dvd/price)
3. Test cumulative adjustments (multiple events over time)
4. Test edge cases (no adjustments, adjustment on first/last date)

## Implementation Steps

1. **Create unit test** (`tests/test_russell1000_adjustments.py`)
   - Test adjustment logic using ETF data
   - Ensure test passes with reference implementation

2. **Create `src/get_russell1000_adjustments.py`**
   - Download adjustment factors from Bloomberg
   - Save to CSV format matching Bloomberg output

3. **Create `src/russell1000_close_at_exchange_auction_adjusted.py`**
   - Read unadjusted prices
   - Apply adjustments using `get_daily_adj()` helper
   - Write adjusted prices

4. **Add DVC stages** to `dvc.yaml`
   - `russell1000_adjustments` - one-time Bloomberg download
   - `russell1000_close_at_exchange_auction_adjusted` - apply adjustments

5. **Run unit tests** to verify correctness

6. **Run DVC pipeline**
   - `dvc repro russell1000_adjustments` (once)
   - `dvc repro russell1000_close_at_exchange_auction_adjusted`

## Key Design Decisions

1. **Separate stages**: Adjustment factor download is separate from application
   - Allows one-time Bloomberg download
   - Adjustment application can be re-run without hitting Bloomberg

2. **Follow existing patterns**: Mirror `get_adjustments.py` and `adr_mid_at_ordinary_auction.py`
   - Proven workflow
   - Consistent with codebase

3. **Use Bloomberg's adjustment factors directly**: Don't recompute from dividends/splits
   - More reliable (handles complex events like rights issues, spinoffs)
   - Matches Bloomberg's official adjusted prices
   - Simpler implementation

4. **One CSV per exchange** for adjusted prices
   - Mirrors unadjusted structure
   - Easy to use in signal generation

## Validation

1. **Unit tests pass** using ETF reference data
2. **Spot-check** specific tickers with known splits/dividends
3. **Sanity checks**:
   - Adjusted prices should be continuous (no jumps on ex-dates)
   - Price * cum_adj should match expected adjusted price
   - Recent prices should have cum_adj ≈ 1.0 (if no recent events)

## Files to Create/Modify

**Create**:
- `src/get_russell1000_adjustments.py`
- `src/russell1000_close_at_exchange_auction_adjusted.py`
- `tests/test_russell1000_adjustments.py`

**Modify**:
- `dvc.yaml` (add two new stages)
