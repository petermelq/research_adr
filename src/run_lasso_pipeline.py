"""
Main pipeline script for LASSO residual prediction.

Runs the full pipeline:
1. Prepare features for all ordinary stocks
2. Train rolling LASSO models
3. Compile results
"""

import sys
import os
from pathlib import Path

__script_dir__ = Path(__file__).parent.absolute()

# Import the pipeline scripts
sys.path.append(str(__script_dir__))


def main():
    """Run the full LASSO pipeline."""
    print("=" * 80)
    print(" " * 20 + "LASSO RESIDUAL PREDICTION PIPELINE")
    print("=" * 80)

    # Step 1: Prepare features
    print("\n" + "=" * 80)
    print("STEP 1: Preparing Features")
    print("=" * 80)

    import prepare_lasso_features
    prepare_lasso_features.main()

    # Step 2: Train models
    print("\n" + "=" * 80)
    print("STEP 2: Training LASSO Models")
    print("=" * 80)

    import train_lasso_models
    train_lasso_models.main()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review training results in: data/processed/models/with_us_stocks/lasso/training_metadata.csv")
    print("  2. Run analysis notebook: notebooks/lasso_russell_analysis.ipynb")
    print("=" * 80)


if __name__ == '__main__':
    main()
