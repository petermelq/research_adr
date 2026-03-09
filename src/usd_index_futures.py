from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adr_strategy_kernel.pipelines.usd_index_futures import convert_index_futures_to_usd, main

__all__ = ["convert_index_futures_to_usd", "main"]


if __name__ == "__main__":
    main()
