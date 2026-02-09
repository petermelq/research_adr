"""
Main pipeline script for Ridge residual prediction.

Runs the full pipeline:
1. Prepare features for all ordinary stocks (if needed)
2. Train rolling Ridge models
3. Compile results
"""

import sys
import os
from pathlib import Path

__script_dir__ = Path(__file__).parent.absolute()

# Import the pipeline scripts
sys.path.append(str(__script_dir__))


def main():
    """Run the full Ridge pipeline."""
    print("=" * 80)
    print(" " * 20 + "RIDGE RESIDUAL PREDICTION PIPELINE")
    print("=" * 80)

    # Check if features already exist
    features_dir = __script_dir__ / '..' / 'data' / 'processed' / 'models' / 'with_us_stocks' / 'features'
    feature_files = list(features_dir.glob('*.parquet')) if features_dir.exists() else []

    if len(feature_files) == 0:
        # Step 1: Prepare features
        print("\n" + "=" * 80)
        print("STEP 1: Preparing Features")
        print("=" * 80)

        import prepare_lasso_features
        prepare_lasso_features.main()
    else:
        print("\n" + "=" * 80)
        print(f"Features already exist ({len(feature_files)} files), skipping preparation")
        print("=" * 80)

    # Step 2: Train models
    print("\n" + "=" * 80)
    print("STEP 2: Training Ridge Models")
    print("=" * 80)

    import train_ridge_models
    train_ridge_models.main()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review training results in: data/processed/models/with_us_stocks/ridge/training_metadata.csv")
    print("  2. Compare with LASSO results")
    print("=" * 80)


if __name__ == '__main__':
    main()
