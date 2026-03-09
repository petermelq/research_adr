from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adr_strategy_kernel.pipelines.adr_mid_at_ordinary_auction import (
    process_adr_mids_efficiently,
    run_adr_mid_at_ordinary_auction,
    main,
)

__all__ = ["process_adr_mids_efficiently", "run_adr_mid_at_ordinary_auction", "main"]


if __name__ == "__main__":
    main()
