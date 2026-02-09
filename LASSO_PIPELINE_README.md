# LASSO Russell 1000 Residual Prediction Pipeline

## Overview

This pipeline predicts ordinary stock return residuals using Russell 1000 returns as features. The model aims to capture cross-market spillover effects beyond what's explained by index beta.

## Pipeline Architecture

```
Ordinary Returns  ──┐
                    ├──> Existing Beta Model ──> Residuals (Target)
Index Returns     ──┘

Russell 1000 Returns ──┐
                       ├──> Residualization ──> Features
Index Returns        ──┘

Features + Target ──> LASSO Model ──> Predictions
```

## Key Features

1. **Residualized Features**: Russell 1000 returns are residualized against the same index used for the ordinary stock, removing common index exposure

2. **Rolling Windows**: Models are trained on 12-month rolling windows with monthly refitting

3. **Hyperparameter Tuning**: Regularization parameter (alpha) is tuned on a validation set (last month of each 12-month window)

4. **Sparse Models**: LASSO selects the most relevant Russell stocks as predictors

5. **Holiday Handling**: Gracefully handles different market calendars by computing multi-day returns when needed

## Data Flow

### Inputs
- **Ordinary prices**: `data/raw/ordinary/ord_PX_LAST_adjust_all.csv`
- **Russell 1000 adjusted prices**: `data/processed/russell1000/close_at_exchange_auction_adjusted/`
- **Index prices**: `data/raw/indices/indices_PX_LAST.csv`
- **Existing betas**: `data/processed/models/ordinary_betas_index_only.csv`
- **Mapping**: `data/raw/adr_info.csv`

### Outputs
- **Features**: `data/processed/models/with_us_stocks/features/{ticker}.parquet`
- **Models**: `data/processed/models/with_us_stocks/lasso/{ticker}/{year}_{month}.pkl`
- **Metadata**: `data/processed/models/with_us_stocks/lasso/training_metadata.csv`

## Usage

### Run Full Pipeline

```bash
# Via DVC (recommended)
dvc repro lasso_train_models

# Or manually
python src/run_lasso_pipeline.py
```

### Run Individual Steps

```bash
# Step 1: Prepare features
python src/prepare_lasso_features.py

# Step 2: Train models
python src/train_lasso_models.py
```

### Analyze Results

```bash
jupyter notebook notebooks/lasso_russell_analysis.ipynb
```

## Configuration

### Index Mapping

The pipeline uses the following mapping from index futures to indices:

- **NH** (Nikkei futures) → **NKY** (Nikkei 225)
- **VG** (EuroStoxx futures) → **SXXP** (STOXX Europe 600)
- **Z** (FTSE futures) → **UKX** (FTSE 100)

This mapping is hardcoded in `src/utils_lasso_residuals.py`.

### Model Parameters

- **Training window**: 11 months
- **Validation window**: 1 month
- **Refitting frequency**: Monthly
- **Rolling beta window**: 60 business days
- **Alpha grid**: `np.logspace(-4, 1, 50)` (50 values from 0.0001 to 10)
- **Feature standardization**: Yes
- **Intercept**: No

### Date Range

- **Start**: 2024-01-02 (from `params.yaml`)
- **End**: 2026-01-30 (from `params.yaml`)

## Model Details

### LASSO Regularization

The LASSO (Least Absolute Shrinkage and Selection Operator) model:

```
minimize: (1/2n) ||y - Xβ||² + α||β||₁
```

- Performs feature selection (many coefficients → 0)
- Prevents overfitting with ~996 features
- Alpha controls regularization strength

### Validation Metric

**Information Coefficient (IC)** = correlation(predictions, actuals)

- IC > 0.05: Good predictive power
- IC > 0.10: Strong predictive power
- Averaged across validation sets for each ticker

## File Structure

```
src/
  utils_lasso_residuals.py      # Utility functions
  prepare_lasso_features.py     # Feature preparation
  train_lasso_models.py          # LASSO training
  run_lasso_pipeline.py          # Main pipeline

tests/
  test_lasso_utils.py            # Unit tests for utilities
  test_lasso_training.py         # Unit tests for training

notebooks/
  lasso_russell_analysis.ipynb   # Analysis notebook

data/processed/models/with_us_stocks/
  features/                      # Prepared features per ticker
  lasso/                         # Trained models per ticker
  training_metadata.csv          # Performance metrics
```

## Testing

Run unit tests:

```bash
pytest tests/test_lasso_utils.py tests/test_lasso_training.py -v
```

Tests cover:
- Return computation with holidays
- Rolling beta residualization
- LASSO model fitting and prediction
- Rolling window generation

## Performance Expectations

Based on the model design:

- **Sparsity**: Expect 5-10% of Russell stocks selected on average
- **IC**: Target average IC > 0.05 across stocks
- **Training time**: ~30 seconds per ticker (with ~24 models per ticker)
- **Total pipeline**: ~30-60 minutes for all ordinary stocks

## Troubleshooting

### Common Issues

1. **Missing ordinary ticker in prices**:
   - Check that ticker exists in `ord_PX_LAST_adjust_all.csv`
   - Verify ticker format matches ADR info

2. **No Russell data for exchange**:
   - Ensure exchange is not XTKS or XASX (excluded)
   - Check that adjusted Russell files exist

3. **Insufficient data for training**:
   - Need at least 12 months of aligned data
   - Check date range and holiday handling

4. **Poor IC performance**:
   - Review feature residualization (may need different index)
   - Check if ordinary stock has enough variance
   - Verify beta model residuals are non-zero

## Extensions

The pipeline is modular and can be extended:

1. **Alternative models**: Replace `LASSOResidualModel` with Ridge, ElasticNet, or other sklearn models
2. **Different features**: Modify `prepare_lasso_features.py` to include volume, order flow, etc.
3. **Custom residualization**: Change rolling beta window or use different index
4. **Alternative validation**: Use cross-validation instead of single validation month

## References

- **LASSO**: Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
- **Information Coefficient**: Common metric in quantitative finance for signal quality
- **Russell 1000**: Broad US equity index representing ~92% of US market cap

## Contact

For questions or issues, refer to the main project documentation or create an issue in the repository.
