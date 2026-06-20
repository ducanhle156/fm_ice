"""Download hourly air temperature from the Iowa Environmental Mesonet (IEM)
ASOS/METAR archive for the nearest airport to each station.

Endpoint (verified 2026-06-20):
  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
  params: station=<id>&data=tmpf&tz=Etc/UTC&format=onlycomma&missing=M&trace=T
          &year1&month1&day1&year2&month2&day2&report_type=3

Notes:
  * IEM throttles to ~1 request/second per IP. We request a whole winter in one
    call, so this is not a concern.
  * data=tmpf is air temperature in Fahrenheit; we also keep relh/dwpf is optional.
    We convert to Celsius on write.
  * report_type=3 selects routine + special; drop to 3 only if you want hourly METARs.

Usage:
  python -m fm_ice.data.download_temperature --station cedarburg --winter 2024-2025
  python -m fm_ice.data.download_temperature --station bismarck --all-winters

Output:
  data/raw/temperature/<station>/<asos>_<winter>_tmp.csv  (datetime_utc, tmpf, tmpc)
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from fm_ice.config import load_yaml, winter_bounds
from fm_ice.http_util import get, session


def fetch_asos(s, base: str, asos: str, start, end) -> str:
    params = {
        "station": asos,
        "data": "tmpf",
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "missing": "M",
        "trace": "T",
        "latlon": "no",
        "direct": "no",
        "report_type": "3",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
    }
    r = get(s, base, params=params)
    return r.text


def parse_asos(text: str) -> list[list]:
    """IEM 'onlycomma' returns: station,valid,tmpf  (valid is UTC 'YYYY-MM-DD HH:MM')."""
    out = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        raw = (row.get("tmpf") or "").strip()
        if raw in ("", "M", "T"):
            continue
        try:
            tmpf = float(raw)
        except ValueError:
            continue
        tmpc = round((tmpf - 32.0) * 5.0 / 9.0, 2)
        out.append([row["valid"], tmpf, tmpc])
    return out


def run(station_key: str, winter: str, out_root: Path, asos_override: str | None) -> None:
    cfg = load_yaml("stations.yaml")
    d = cfg["defaults"]
    st = cfg["stations"][station_key]
    asos = asos_override or st["asos_station"]
    start, end = winter_bounds(winter, d["season_start_md"], d["season_end_md"])

    s = session()
    text = fetch_asos(s, d["iem_asos_base"], asos, start, end)
    rows = parse_asos(text)

    out_dir = out_root / station_key
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{asos}_{winter}_tmp.csv"
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["datetime_utc", "tmpf", "tmpc"])
        w.writerows(rows)
    print(f"[temp] {station_key} {winter} (ASOS {asos}): {len(rows)} obs -> {dst}")
    if not rows:
        print("  WARNING: 0 rows. Check the ASOS id against "
              "https://mesonet.agron.iastate.edu/request/download.phtml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--winter")
    ap.add_argument("--all-winters", action="store_true")
    ap.add_argument("--asos", help="override the ASOS id in stations.yaml")
    ap.add_argument("--out", default="data/raw/temperature")
    args = ap.parse_args()
    cfg = load_yaml("stations.yaml")
    winters = cfg["stations"][args.station]["winters"] if args.all_winters else [args.winter]
    if not winters or winters == [None]:
        raise SystemExit("Provide --winter YYYY-YYYY or --all-winters")
    for w in winters:
        run(args.station, w, Path(args.out), args.asos)


if __name__ == "__main__":
    main()
