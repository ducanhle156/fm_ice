"""Download HIVIS camera images for a station and winter.

Flow (per the NIMS v0 OpenAPI, verified 2026-06-20):
  1. GET /cameras?camId=<id>     -> read overlayDir / smallDir / thumbDir (S3 base).
  2. GET /listFiles?camId=<id>&after=<iso>&before=<iso>&recent=false&limit=<n>
                                 -> list of filenames in the window.
  3. For each filename, download  <baseDir> + <filename>  from public S3.

Filenames look like:  <camId>___YYYY-MM-DDTHH-MM-SSZ.jpg   (UTC, hourly).

Usage:
  python -m fm_ice.data.download_images --station cedarburg --winter 2024-2025
  python -m fm_ice.data.download_images --station bismarck --winter 2025-2026 --size small
  python -m fm_ice.data.download_images --station cedarburg --all-winters --dry-run

Outputs:
  data/raw/images/<station>/<camId>/<filename>.jpg
  data/raw/images/<station>/<camId>/_manifest.csv   (filename, timestamp_utc, url, bytes)

Notes:
  * No API key is required for low-volume use. If you hit OVER_RATE_LIMIT, set
    USGS_API_KEY (see fm_ice.http_util.session) from https://api.waterdata.usgs.gov/signup/.
  * Use --dry-run first to see counts and estimated volume before pulling GBs.
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import date, datetime, timezone
from pathlib import Path

from fm_ice.config import load_yaml, winter_bounds
from fm_ice.http_util import get, session

SIZE_TO_DIR = {"overlay": "overlayDir", "small": "smallDir", "thumb": "thumbDir"}
FNAME_RE = re.compile(r"___(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})Z\.jpg$")


def parse_ts(filename: str) -> datetime | None:
    """Pull the UTC timestamp out of a NIMS filename."""
    m = FNAME_RE.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)


def nims_date(d: date, end: bool = False) -> str:
    """NIMS-standard date string accepted by /listFiles after/before."""
    return f"{d.isoformat()}T{'23-59-59' if end else '00-00-00'}Z"


def fetch_camera(s, api_base: str, cam_id: str) -> dict:
    r = get(s, f"{api_base}/cameras", params={"camId": cam_id})
    arr = r.json()
    if not arr:
        raise SystemExit(f"No camera found for camId={cam_id!r}")
    return arr[0]


def list_files(s, api_base: str, cam_id: str, start: date, end: date, limit: int = 50000) -> list[str]:
    r = get(
        s,
        f"{api_base}/listFiles",
        params={
            "camId": cam_id,
            "after": nims_date(start),
            "before": nims_date(end, end=True),
            "recent": "false",   # oldest-first
            "limit": limit,
        },
    )
    return r.json()


def download_window(station_key: str, cam_role: str, winter: str, size: str,
                    out_root: Path, dry_run: bool) -> None:
    stations = load_yaml("stations.yaml")
    defaults = stations["defaults"]
    st = stations["stations"][station_key]
    cam = st["cameras"][cam_role]
    cam_id = cam["cam_id"]
    api_base = defaults["hivis_api_base"]

    start, end = winter_bounds(winter, defaults["season_start_md"], defaults["season_end_md"])
    # Do not request before the camera existed.
    arch_start = datetime.strptime(cam["archive_start"], "%Y-%m-%d").date()
    start = max(start, arch_start)
    if start > end:
        print(f"[skip] {station_key}/{cam_role} {winter}: camera not yet online in this winter.")
        return

    s = session()
    meta = fetch_camera(s, api_base, cam_id)
    base_dir = meta[SIZE_TO_DIR[size]]
    files = list_files(s, api_base, cam_id, start, end)
    files = [f for f in files if parse_ts(f) is not None]
    files.sort()

    approx_mb = len(files) * (0.27 if size == "overlay" else 0.06 if size == "small" else 0.01)
    print(f"[{station_key}/{cam_role}] {winter}: {len(files)} images "
          f"({start}..{end}), ~{approx_mb:.0f} MB at size={size}")
    if dry_run:
        return

    out_dir = out_root / station_key / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "_manifest.csv"
    new_rows = []
    for i, fn in enumerate(files, 1):
        dst = out_dir / fn
        url = base_dir + fn
        if not dst.exists():
            r = get(s, url, stream=True)
            with open(dst, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        ts = parse_ts(fn)
        new_rows.append([fn, ts.isoformat(), url, dst.stat().st_size])
        if i % 250 == 0:
            print(f"  ... {i}/{len(files)}")

    write_header = not manifest.exists()
    with open(manifest, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["filename", "timestamp_utc", "url", "bytes"])
        w.writerows(new_rows)
    print(f"[done] {len(new_rows)} files -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True, help="key in configs/stations.yaml, e.g. cedarburg")
    ap.add_argument("--cam-role", default="primary", help="camera role within the station (default: primary)")
    ap.add_argument("--winter", help="e.g. 2024-2025")
    ap.add_argument("--all-winters", action="store_true", help="download every winter listed for the station")
    ap.add_argument("--size", default="overlay", choices=list(SIZE_TO_DIR), help="overlay=full, small=720px, thumb")
    ap.add_argument("--out", default="data/raw/images", help="output root")
    ap.add_argument("--dry-run", action="store_true", help="print counts and volume only")
    args = ap.parse_args()

    stations = load_yaml("stations.yaml")
    winters = stations["stations"][args.station]["winters"] if args.all_winters else [args.winter]
    if not winters or winters == [None]:
        raise SystemExit("Provide --winter YYYY-YYYY or --all-winters")
    for w in winters:
        download_window(args.station, args.cam_role, w, args.size, Path(args.out), args.dry_run)


if __name__ == "__main__":
    main()
