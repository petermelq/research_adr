# Bloomberg API Issue - Adjustment Factors

## Current Status

✅ **Infrastructure Complete**:
- Unit tests passing (3/3 core logic tests)
- Scripts created and tested
- DVC stages configured
- Adjusted price files generated (13 exchange CSVs)

⚠️ **Bloomberg API Returning Empty Results**:
- `blp.bds()` with `EQY_DVD_ADJUST_FACT` returns empty DataFrame
- Tested with multiple approaches - all return empty
- This prevents downloading actual adjustment factors

## What Was Created

Since Bloomberg returned no data, the system created:
- Empty adjustment factors file: `data/processed/russell1000/adjustment_factors.csv`
- "Adjusted" price files that currently match unadjusted prices (no adjustments applied)

**Files Created**:
```
data/processed/russell1000/
├── adjustment_factors.csv (empty - needs Bloomberg data)
└── close_at_exchange_auction_adjusted/
    ├── XLON.csv (467 rows x 996 cols)
    ├── XAMS.csv (469 rows x 996 cols)
    ├── XPAR.csv (469 rows x 996 cols)
    ├── XETR.csv (467 rows x 996 cols)
    ├── XMIL.csv (467 rows x 996 cols)
    ├── XBRU.csv (469 rows x 996 cols)
    ├── XMAD.csv (469 rows x 996 cols)
    ├── XHEL.csv (462 rows x 996 cols)
    ├── XDUB.csv (467 rows x 996 cols)
    ├── XOSL.csv (462 rows x 996 cols)
    ├── XSTO.csv (454 rows x 996 cols)
    ├── XSWX.csv (460 rows x 996 cols)
    └── XCSE.csv (462 rows x 996 cols)
```

## Bloomberg API Tests Attempted

### Test 1: Using `blp.bds()` directly
```python
blp.bds('AAPL US Equity', 'EQY_DVD_ADJUST_FACT',
        Corporate_Actions_Filter='NORMAL_CASH|ABNORMAL_CASH|CAPITAL_CHANGE')
# Result: Empty DataFrame
```

### Test 2: Using `blp.dividend()`
```python
blp.dividend(['AAPL US Equity'], start_date='2020-01-01',
             end_date='2026-01-30', typ='all', timeout=10000)
# Result: Empty DataFrame
```

### Test 3: Basic connectivity test
```python
blp.bdp(['AAPL US Equity'], 'PX_LAST')
# Result: Error - "No columns to parse from file"
```

## Possible Causes

1. **Bloomberg Terminal Not Running**
   - linux_xbbg requires Bloomberg Terminal to be running on Windows side
   - Check if Terminal is logged in and active

2. **Date Range Issue**
   - Current range: 2024-01-02 to 2026-01-30
   - Maybe Bloomberg needs historical data request differently

3. **API Configuration**
   - linux_xbbg might need specific configuration
   - Check if Windows-side script is accessible

4. **Field/Filter Issue**
   - `EQY_DVD_ADJUST_FACT` field might require different parameters
   - Filter syntax might be incorrect

## Next Steps

### Option 1: Debug Bloomberg Connection (Recommended)
1. **Verify Bloomberg Terminal is running** on Windows side
2. **Test basic Bloomberg query** to confirm connectivity:
   ```python
   from linux_xbbg import blp
   # Try simplest possible query
   result = blp.bdh(['AAPL US Equity'], ['PX_LAST'], '2024-01-01', '2024-01-31')
   print(result)
   ```

3. **If basic queries work**, try adjustment factors with simpler parameters:
   ```python
   # Try without date filtering first
   result = blp.bds(['AAPL US Equity'], 'EQY_DVD_ADJUST_FACT')
   ```

4. **Check your working example code** - Compare with the script you showed:
   ```python
   # From your example:
   div_df = utils.bds(f'{ticker} US Equity', 'EQY_DVD_ADJUST_FACT',
                      Corporate_Actions_Filter='NORMAL_CASH|ABNORMAL_CASH|CAPITAL_CHANGE')
   ```
   - Is `utils.bds()` a wrapper that does something different?
   - Check if there's a utils module with a `bds` function

### Option 2: Use Alternative Data Source
- Download adjustment factors manually from Bloomberg Terminal
- Export to CSV with required format:
  ```
  ticker,adjustment_date,adjustment_factor
  AAPL US Equity,2024-08-07,0.5
  ```

### Option 3: Use Existing Workflow
- Run the existing `get_adjustments.py` script which uses `blp.dividend()`:
  ```bash
  python src/get_adjustments.py \
    data/processed/russell1000/close_at_exchange_auction/XLON.csv \
    data/processed/russell1000/adjustment_factors_temp.csv \
    --symbol_suffix=" US Equity"
  ```
- This computes adjustment factors from dividend data + prices

## To Re-run Once Bloomberg Works

Once you get Bloomberg returning data:

1. **Download adjustment factors**:
   ```bash
   python src/get_russell1000_adjustments.py
   # OR
   dvc repro russell1000_adjustments
   ```

2. **Apply adjustments**:
   ```bash
   python src/russell1000_close_at_exchange_auction_adjusted.py
   # OR
   dvc repro russell1000_close_at_exchange_auction_adjusted
   ```

3. **Verify results**:
   ```python
   import pandas as pd

   # Compare unadjusted vs adjusted
   unadj = pd.read_csv('data/processed/russell1000/close_at_exchange_auction/XLON.csv',
                       index_col=0)
   adj = pd.read_csv('data/processed/russell1000/close_at_exchange_auction_adjusted/XLON.csv',
                     index_col=0)

   # Should see differences for stocks with splits/dividends
   print((adj / unadj).describe())
   ```

## Current State

For now, you have:
- ✅ Complete infrastructure (scripts, tests, DVC)
- ✅ "Adjusted" files (currently matching unadjusted since no Bloomberg data)
- ✅ Ready to re-run once Bloomberg connectivity is fixed

The adjusted files can be used as-is for testing your signal generation pipeline, knowing they currently have no actual adjustments applied.
