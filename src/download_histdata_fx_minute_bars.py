import argparse
import io
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set
from urllib.parse import urljoin

import pandas as pd
import requests


BASE_URL = "https://www.histdata.com"
DOWNLOAD_ROOT = (
    "https://www.histdata.com/download-free-forex-historical-data/"
    "?/ascii/1-minute-bar-quotes"
)
EXPECTED_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and stitch HistData 1-minute FX bars."
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="FX pairs to download, e.g. USDNOK USDSEK USDDKK",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="First year to download (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=pd.Timestamp.utcnow().year,
        help="Last year to download (inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/currencies/minute_bars",
        help="Directory for <PAIR>_full_1min.txt outputs.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Delay between HistData requests.",
    )
    return parser.parse_args()


def _extract_hidden_fields(html: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    required = ("tk", "date", "datemonth", "platform", "timeframe", "fxpair")
    for key in required:
        pattern = (
            rf"<input[^>]+(?:id|name)=[\"']{key}[\"'][^>]*value=[\"']([^\"']+)[\"']"
            rf"|<input[^>]+value=[\"']([^\"']+)[\"'][^>]*(?:id|name)=[\"']{key}[\"']"
        )
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if not match:
            raise RuntimeError(f"Missing hidden field '{key}'")
        fields[key] = next(group for group in match.groups() if group is not None)
    return fields


def _collect_month_links(html: str, pair: str, year: int) -> List[str]:
    suffix = f"/ascii/1-minute-bar-quotes/{pair.lower()}/{year}"
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    month_urls: Set[str] = set()
    for href in links:
        abs_url = urljoin(BASE_URL, href)
        if suffix + "/" in abs_url and abs_url.startswith(BASE_URL):
            month_urls.add(abs_url.rstrip("/"))
    return sorted(month_urls)


def _download_zip_bytes(session: requests.Session, final_url: str) -> bytes:
    response = session.get(final_url, timeout=60)
    response.raise_for_status()
    fields = _extract_hidden_fields(response.text)
    zip_resp = session.post(
        f"{BASE_URL}/get.php",
        data=fields,
        headers={"Origin": BASE_URL, "Referer": final_url},
        timeout=300,
    )
    zip_resp.raise_for_status()
    return zip_resp.content


def _iter_zip_payloads(raw_zip: bytes) -> Iterable[bytes]:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            payload = zf.read(member)
            if not payload:
                continue
            if member.filename.lower().endswith(".zip"):
                yield from _iter_zip_payloads(payload)
            else:
                yield payload


def _read_zip_csv_frames(zip_bytes: bytes) -> Iterable[pd.DataFrame]:
    line_pattern = re.compile(
        r"^(\d{8})\s+(\d{2}:?\d{2}:?\d{2})[;,]([0-9eE+.\-]+)[;,]([0-9eE+.\-]+)[;,]([0-9eE+.\-]+)[;,]([0-9eE+.\-]+)[;,]([0-9eE+.\-]+)"
    )
    for payload in _iter_zip_payloads(zip_bytes):
        text = payload.decode("utf-8", errors="ignore")
        records: List[List[str]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("<"):
                continue
            match = line_pattern.match(line)
            if not match:
                continue
            records.append(list(match.groups()))
        if not records:
            continue
        yield pd.DataFrame(records, columns=EXPECTED_COLUMNS)


def _inspect_zip(zip_bytes: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [m.filename for m in zf.infolist() if not m.is_dir()]
            print(f"ZIP debug members: {members[:5]}")
            for member in zf.infolist():
                if member.is_dir():
                    continue
                payload = zf.read(member)
                if member.filename.lower().endswith(".zip"):
                    print(f"ZIP debug nested zip: {member.filename} ({len(payload)} bytes)")
                    return
                text = payload.decode("utf-8", errors="ignore")
                sample = next((ln for ln in text.splitlines() if ln.strip()), "")
                print(f"ZIP debug sample line ({member.filename}): {sample[:200]}")
                return
    except Exception as exc:
        print(f"ZIP debug failed: {exc}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].astype(str).str.replace(r"[^0-9]", "", regex=True).str[-8:]
    out["time"] = out["time"].astype(str).str.strip()
    out["time"] = out["time"].str.replace(r"[^0-9]", "", regex=True)
    out["time"] = out["time"].str.zfill(6).str.replace(
        r"^(\d{2})(\d{2})(\d{2})$", r"\1:\2:\3", regex=True
    )
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "time", "open", "high", "low", "close", "volume"])
    out = out[out["date"].str.len() == 8]
    out["timestamp"] = pd.to_datetime(
        out["date"] + " " + out["time"], format="%Y%m%d %H:%M:%S", errors="coerce"
    )
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates(["date", "time"], keep="last")
    return out[EXPECTED_COLUMNS]


def _download_pair(
    session: requests.Session,
    pair: str,
    start_year: int,
    end_year: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        year_url = f"{DOWNLOAD_ROOT}/{pair.lower()}/{year}"
        year_resp = session.get(year_url, timeout=60)
        if year_resp.status_code >= 400:
            print(f"[{pair}] Skipping year {year}: HTTP {year_resp.status_code}")
            continue
        month_urls = _collect_month_links(year_resp.text, pair, year)
        explicit_month_urls: List[str] = []
        if year == end_year:
            explicit_month_urls = [f"{year_url}/{month}" for month in range(1, 13)]
        final_urls = list(dict.fromkeys(month_urls + explicit_month_urls + [year_url]))
        for final_url in final_urls:
            print(f"[{pair}] Downloading {final_url}")
            try:
                zip_bytes = _download_zip_bytes(session, final_url)
                extracted = list(_read_zip_csv_frames(zip_bytes))
                if not extracted:
                    print(f"[{pair}] No rows parsed for {final_url}")
                    _inspect_zip(zip_bytes)
                frames.extend(extracted)
            except Exception as exc:
                print(f"[{pair}] Failed {final_url}: {exc}")
                continue
            time.sleep(sleep_seconds)
    if not frames:
        raise RuntimeError(f"No data downloaded for {pair}")
    return _normalize(pd.concat(frames, ignore_index=True))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        for pair in args.pairs:
            pair = pair.upper()
            frame = _download_pair(
                session=session,
                pair=pair,
                start_year=args.start_year,
                end_year=args.end_year,
                sleep_seconds=args.sleep_seconds,
            )
            output_path = output_dir / f"{pair}_full_1min.txt"
            frame.to_csv(output_path, index=False, header=False)
            print(f"[{pair}] Wrote {len(frame):,} rows to {output_path}")


if __name__ == "__main__":
    main()
