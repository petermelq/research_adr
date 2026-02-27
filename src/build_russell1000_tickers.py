from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    in_path = root / "data" / "raw" / "historical_russell_1000.csv"
    out_path = root / "data" / "raw" / "russell1000_tickers.csv"

    df = pd.read_csv(in_path, usecols=["id"])
    tickers = (
        df["id"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
        .str.upper()
    )
    # Keep standard symbol forms expected by the FRD downloader.
    tickers = tickers[tickers.str.match(r"^[A-Z][A-Z0-9./-]*$")]
    tickers = pd.Index(tickers.unique()).sort_values()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(out_path, index=False)
    print(f"Saved {len(tickers)} tickers to {out_path}")


if __name__ == "__main__":
    main()
