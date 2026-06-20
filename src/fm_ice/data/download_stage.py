"""Download river stage (gage height, parameter 00065) from the USGS NWIS
Instantaneous Values service.

Endpoint (verified 2026-06-20):
  https://waterservices.usgs.gov/nwis/iv/?format=json&sites=<id>&parameterCd=00065
        &startDT=YYYY-MM-DD&endDT=YYYY-MM-DD&siteStatus=all

Stage is the physics-based, label-free reference: freeze-up produces a
characteristic backwater rise in stage that is independent of the camera.
Discharge (00060) is also pulled because USGS ice qualifiers live on it
(see download_ice_flags.py).

Usage:
  python -m fm_ice.data.download_stage --station cedarburg --winter 2024-2025
  python -m fm_ice.data.download_stage --station bismarck --all-winters

Output:
  data/raw/stage/<station>/<nwisId>_<winter>_iv.csv   (datetime_utc, value, param, qualifiers)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fm_ice.config import load_yaml, winter_bounds
from fm_ice.http_util import get, session


def fetch_iv(s, base: str, site: str, params_cd: str, start, end) -> dict:
    r = get(s, base, params={
        "format": "json",
        "sites": site,
        "parameterCd": params_cd,
        "startDT": start.isoformat(),
        "endDT": end.isoformat(),
        "siteStatus": "all",
    })
    return r.json()


def rows_from_iv(payload: dict) -> list[list]:
    out = []
    for ts in payload.get("value", {}).get("timeSeries", []):
        var = ts["variable"]["variableCode"][0]["value"]   # e.g. 00065
        for block in ts["values"]:
            for v in block["value"]:
                quals = "|".join(v.get("qualifiers", []))
                out.append([v["dateTime"], v["value"], var, quals])
    return out


def run(station_key: str, winter: str, out_root: Path) -> None:
    cfg = load_yaml("stations.yaml")
    d = cfg["defaults"]
    st = cfg["stations"][station_key]
    site = st["nwis_id"]
    start, end = winter_bounds(winter, d["season_start_md"], d["season_end_md"])

    s = session()
    payload = fetch_iv(s, d["nwis_iv_base"], site, "00065,00060", start, end)
    rows = rows_from_iv(payload)

    out_dir = out_root / station_key
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{site}_{winter}_iv.csv"
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["datetime", "value", "param", "qualifiers"])
        w.writerows(rows)
    print(f"[stage] {station_key} {winter}: {len(rows)} rows -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--winter")
    ap.add_argument("--all-winters", action="store_true")
    ap.add_argument("--out", default="data/raw/stage")
    args = ap.parse_args()
    cfg = load_yaml("stations.yaml")
    winters = cfg["stations"][args.station]["winters"] if args.all_winters else [args.winter]
    if not winters or winters == [None]:
        raise SystemExit("Provide --winter YYYY-YYYY or --all-winters")
    for w in winters:
        run(args.station, w, Path(args.out))


if __name__ == "__main__":
    main()
