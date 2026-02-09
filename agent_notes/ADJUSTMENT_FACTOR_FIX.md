# Adjustment Factor Fix Summary

## Problem
The adjustment logic was not correctly handling Bloomberg's adjustment factors, resulting in a max difference of 586.90 between our adjusted prices and Bloomberg's official adjusted prices.

## Root Cause
Bloomberg's `EQY_DVD_ADJUST_FACT` field includes an `adjustment_factor_operator_type` that specifies how to apply the factor:
- **Type 1.0**: Divide by the factor (used for reverse splits)
- **Type 2.0**: Multiply by the factor (used for dividends and forward splits)

Our original code always multiplied by the adjustment factor, which was incorrect for reverse splits.

## Example
SONY had a 1:5 reverse split on 2024-10-09 with:
- `adjustment_factor = 5.0`
- `adjustment_factor_operator_type = 1.0` (divide)

Our code was multiplying by 5.0 (making pre-split prices 5x higher), when it should have been dividing by 5.0 (making pre-split prices 5x lower to match post-split scale).

## Solution
Updated the adjustment logic to check `adjustment_factor_operator_type`:

```python
if 'adjustment_factor_operator_type' in adj_group.columns:
    mask = adj_group['adjustment_factor_operator_type'] == 1.0
    adj_group.loc[mask, 'adjustment_factor'] = 1.0 / adj_group.loc[mask, 'adjustment_factor']
```

This inverts the factor for Type 1.0 operations, converting divide operations to multiply operations (since all our code uses multiplication).

## Files Updated

1. **tests/test_russell1000_adjustments.py**
   - Updated `get_daily_adj()` to handle operator_type
   - Updated `apply_adjustments()` to reset column names after pivot
   - Adjusted tolerance to rtol=1e-4, atol=1e-4 (0.01% relative tolerance)
   - All 4 tests now pass

2. **src/russell1000_close_at_exchange_auction_adjusted.py**
   - Updated `get_daily_adj()` to handle operator_type
   - Production script ready to use

3. **src/get_russell1000_adjustments.py**
   - Simplified to use only `blp.bds()` with `EQY_DVD_ADJUST_FACT`
   - Removed complex logic that tried to compute adjustments from dividend data
   - Now downloads all adjustment factors directly from Bloomberg BDS
   - Includes `adjustment_factor_operator_type` and `adjustment_factor_flag` in output

## Test Results
- Bloomberg adjustment test: **PASSED** (max diff: 0.00005, down from 586.90)
- Split adjustment test: **PASSED**
- Dividend adjustment test: **PASSED**
- Cumulative adjustments test: **PASSED**

## Next Steps
1. Run `python src/get_russell1000_adjustments.py` to download full Russell 1000 adjustment factors
2. Run `python src/russell1000_close_at_exchange_auction_adjusted.py` to create adjusted price files
3. Verify the adjusted prices look reasonable

## Data Columns
The adjustment factors CSV now includes:
- `ticker`: Stock ticker with " US Equity" suffix
- `adjustment_date`: Ex-date of the corporate action
- `adjustment_factor`: The adjustment factor value
- `adjustment_factor_operator_type`: 1.0 (divide) or 2.0 (multiply)
- `adjustment_factor_flag`: 1.0 (cash dividend) or 3.0 (capital change/split)
