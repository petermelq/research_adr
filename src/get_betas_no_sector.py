from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adr_strategy_kernel.pipelines.index_only_betas import compute_index_only_betas, main

__all__ = ["compute_index_only_betas", "main"]


if __name__ == "__main__":
    main()
