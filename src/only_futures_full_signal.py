from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adr_strategy_kernel.pipelines.only_futures_full_signal import (
    prepare_adr_baseline,
    run_only_futures_full_signal,
    main,
)

__all__ = ["prepare_adr_baseline", "run_only_futures_full_signal", "main"]


if __name__ == "__main__":
    main()
