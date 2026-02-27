import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invert FX minute bar files (e.g. USDJPY -> JPYUSD).")
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Mapping in the form INPUT:OUTPUT (e.g. USDNOK:NOKUSD). Repeat for multiple pairs.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/currencies/minute_bars",
        help="Directory containing <PAIR>_full_1min.txt files.",
    )
    return parser.parse_args()


def invert_file(input_path: Path, output_path: Path) -> None:
    bars = pd.read_csv(
        input_path,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    bars[["open", "close"]] = 1.0 / bars[["open", "close"]]
    bars[["high", "low"]] = 1.0 / bars[["low", "high"]]
    bars.to_csv(output_path, index=False, header=False)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    for mapping in args.pair:
        if ":" not in mapping:
            raise ValueError(f"Invalid --pair mapping: {mapping}")
        input_pair, output_pair = mapping.upper().split(":", 1)
        input_path = data_dir / f"{input_pair}_full_1min.txt"
        output_path = data_dir / f"{output_pair}_full_1min.txt"
        invert_file(input_path, output_path)
        print(f"Inverted {input_pair} -> {output_pair}: {output_path}")


if __name__ == "__main__":
    main()
