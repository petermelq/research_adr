from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adr_strategy_kernel.pipelines.closing_domestic_prices import (
    convert_ordinary_closes_to_usd,
    get_sh_per_adr,
    main,
)

__all__ = ["convert_ordinary_closes_to_usd", "get_sh_per_adr", "main"]


if __name__ == "__main__":
    main()
