from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adr_strategy_kernel.pipelines.build_aligned_index_prices import build_aligned_index_prices, main

__all__ = ["build_aligned_index_prices", "main"]


if __name__ == "__main__":
    main()
