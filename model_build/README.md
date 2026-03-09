`model_build/` is a nested DVC subpipeline for daily production-style model assembly from a single artifact directory.

The only user-managed input is [model_artifact](./model_artifact), which carries the strategy universe snapshot, the rolling-window parameters, the target trade date, and the kernel commit metadata.

`market_data_start_date` and `intraday_start_date` are not part of the artifact schema anymore. `model_build` derives them from `trade_date`, the configured lookbacks, and the exchange calendars in the artifact universe snapshot.

Usage:

1. Refresh the imported intraday inputs:
   `python model_build/scripts/update_imports.py --artifact-dir model_build/model_artifact`
2. Reproduce the subpipeline:
   `dvc repro model_build/dvc.yaml`

Current local raw snapshots in this worktree appear to stop at `2026-02-13`. If the artifact trade date stays at `2026-03-09`, `validate_imports` will fail until the source DVC repo or imported raw files include March 9, 2026 data.

Outputs:

- `model_build/data/processed/models/ordinary_betas_index_only.csv`
- `model_build/data/processed/current_covariance/current_covariance.csv`
- `model_build/data/processed/current_covariance/current_covariance_metadata.json`

The covariance stage reproduces the residual covariance used by `hedged_single_time_ADR` for the artifact trade date. It uses the same fixed-time signal, ADR/ETF fixed-time mids, close prices, and hedge ratios as the backtest path.
