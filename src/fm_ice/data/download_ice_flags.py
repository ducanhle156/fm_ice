"""Download the USGS operational ice reference (the 'ice flag').

USGS records ice-affected periods as a data qualifier on the discharge record.
Crucially, the encoding depends on the approval state of the record:
  * Provisional data (qualifier 'P') carries the literal 'Ice' qualifier on
    ice-affected days.
  * Approved data (qualifier 'A') DROPS the 'Ice' string. When a river goes under
    ice the stage-discharge rating breaks, so USGS publishes the daily discharge
    as ESTIMATED (qualifier 'e'). On the approved record that 'e' IS the ice
    signal -- verified 2026-06-20: the estimated days cluster Nov-Mar exactly as
    an ice season should, while the literal 'Ice' string appears only on the
    current provisional winter.

So matching only 'Ice' reports zero ice for every approved winter -- i.e. for
exactly the older winters used for training/testing. We therefore treat the
cold-season 'e' (estimated) qualifier as the ice-affected marker, and keep the
explicit 'Ice' string in a separate column so the two can be audited apart.

Endpoint (verified 2026-06-20):
  https://waterservices.usgs.gov/nwis/dv/?format=json&sites=<id>&parameterCd=00060
        &statCd=00003&startDT=YYYY-MM-DD&endDT=YYYY-MM-DD

We emit a per-day table with the combined ice_flag plus the two component
columns and the raw qualifier codes so you can audit. Onset is the first
sustained ice day; breakup is the last. The event-extraction logic lives in
fm_ice.baselines (this script only fetches).

Caveat: 'e' (estimated) is not exclusively ice -- discharge can be estimated for
other reasons -- but within the downloaded cold-season window it is dominated by
ice. Daily ice qualifiers are also coarse (1-day resolution) and can lag. Treat
them as one of several references, alongside stage breakpoints. Cross-check a
sample against the annual USGS station analysis / data grades before trusting.

Usage:
  python -m fm_ice.data.download_ice_flags --station cedarburg --all-winters

Output:
  data/raw/ice_flags/<station>/<nwisId>_<winter>_iceflag.csv
    columns: date, ice_flag, ice_explicit, estimated, qualifiers
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fm_ice.config import load_yaml, winter_bounds
from fm_ice.http_util import get, session


def fetch_dv(s, base: str, site: str, start, end) -> dict:
    r = get(s, base, params={
        "format": "json",
        "sites": site,
        "parameterCd": "00060",
        "statCd": "00003",   # daily mean
        "startDT": start.isoformat(),
        "endDT": end.isoformat(),
        "siteStatus": "all",
    })
    return r.json()


def rows_from_dv(payload: dict) -> list[list]:
    """One row per day: combined ice_flag plus its two components.

    ice_explicit -- the literal 'Ice' qualifier (present only on provisional data).
    estimated    -- the 'e' qualifier; on approved records this is the ice signal.
    ice_flag     -- ice_explicit OR estimated (the usable ice-affected marker).
    """
    out = []
    for ts in payload.get("value", {}).get("timeSeries", []):
        for block in ts["values"]:
            for v in block["value"]:
                raw = v.get("qualifiers", [])
                quals = [q.strip().upper() for q in raw]
                ice_explicit = any("ICE" in q for q in quals)
                estimated = any(q == "E" for q in quals)
                ice_flag = ice_explicit or estimated
                date_str = v["dateTime"][:10]
                out.append([
                    date_str,
                    int(ice_flag),
                    int(ice_explicit),
                    int(estimated),
                    "|".join(raw),
                ])
    return out


def run(station_key: str, winter: str, out_root: Path) -> None:
    cfg = load_yaml("stations.yaml")
    d = cfg["defaults"]
    st = cfg["stations"][station_key]
    site = st["nwis_id"]
    start, end = winter_bounds(winter, d["season_start_md"], d["season_end_md"])

    s = session()
    payload = fetch_dv(s, d["nwis_dv_base"], site, start, end)
    rows = rows_from_dv(payload)
    n_ice = sum(r[1] for r in rows)
    n_explicit = sum(r[2] for r in rows)
    n_estimated = sum(r[3] for r in rows)

    out_dir = out_root / station_key
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{site}_{winter}_iceflag.csv"
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "ice_flag", "ice_explicit", "estimated", "qualifiers"])
        w.writerows(rows)
    print(f"[iceflag] {station_key} {winter}: {len(rows)} days, {n_ice} ice-flagged "
          f"({n_explicit} explicit 'Ice', {n_estimated} estimated 'e') -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--winter")
    ap.add_argument("--all-winters", action="store_true")
    ap.add_argument("--out", default="data/raw/ice_flags")
    args = ap.parse_args()
    cfg = load_yaml("stations.yaml")
    winters = cfg["stations"][args.station]["winters"] if args.all_winters else [args.winter]
    if not winters or winters == [None]:
        raise SystemExit("Provide --winter YYYY-YYYY or --all-winters")
    for w in winters:
        run(args.station, w, Path(args.out))


if __name__ == "__main__":
    main()
