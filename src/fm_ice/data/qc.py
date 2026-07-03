"""Per-frame quality control: night, glare, occlusion, and a usable fraction.

River-ice imagery is only informative in usable daylight. Night frames, sun
glare blow-out, and a snow/rain/fog-covered lens carry no ice signal and would
poison both the segmentation baseline and the embedding head. This module scores
every frame once, caches the result, and lets ``assemble_clips`` aggregate a
per-clip ``valid_fraction``.

Two independent signals:
  * Geometry -- solar elevation from (lat, lon, UTC time) via the NOAA solar
    position algorithm. ``is_night`` = sun below the horizon. No pixels needed,
    so it is exact and cheap.
  * Photometry -- brightness/contrast/blown-out/dark statistics on a
    quarter-resolution decode of the JPEG (cv2 IMREAD_REDUCED_COLOR_4). These
    catch glare and lens occlusion that the clock cannot.

A frame is ``usable`` when it is daytime AND not glared AND not occluded AND not
effectively black. Thresholds are module constants, documented and easy to tune;
the distribution is printed at build time so they can be sanity-checked per site.

Outputs (cached): data/interim/<station>/<cam_id>_frame_qc.csv
  filename, timestamp_utc, solar_elev_deg, is_night, brightness, contrast,
  overexposed_frac, dark_frac, glare, occluded, usable
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
    cv2.setNumThreads(0)   # we parallelize at the frame level ourselves
except ImportError:  # pragma: no cover
    cv2 = None

from fm_ice.config import load_yaml, winter_bounds


def to_hourly(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    """Collapse to at most one frame per clock hour: the one closest to the top
    of the hour. Some camera-winters (e.g. Cedarburg 2022-2023) capture in bursts
    a few minutes apart instead of the documented hourly cadence, which would push
    clips well past the fixed frame count the V-JEPA checkpoint requires and make
    valid_fraction exceed 1. Enforcing hourly here makes every station-winter
    uniform without changing the clip definition."""
    d = df.copy()
    # Round (not floor) to the NEAREST hour: a frame at HH:58 belongs to (HH+1):00,
    # and its distance is measured to that hour, so the frame truly closest to each
    # top-of-hour wins. floor() would misbucket and misrank cross-boundary frames.
    hour = d[ts_col].dt.round("h")
    dist = (d[ts_col] - hour).abs()
    d = d.assign(_hour=hour, _dist=dist).sort_values(["_hour", "_dist"])
    d = d.drop_duplicates("_hour", keep="first").drop(columns=["_hour", "_dist"])
    return d.sort_values(ts_col).reset_index(drop=True)


# --- Photometric thresholds (0..1 brightness/contrast scale) ----------------
OVEREXPOSED_LEVEL = 245 / 255       # a pixel this bright is blown out
DARK_LEVEL = 15 / 255               # a pixel this dim carries no detail
GLARE_OVEREXP_FRAC = 0.10           # >10% blown-out area = glare
GLARE_BRIGHT = 0.85                 # very bright ...
GLARE_LOW_CONTRAST = 0.06           # ... and flat = white-out glare
OCCLUDED_CONTRAST = 0.04            # near-uniform field = lens covered/fogged
BLACK_BRIGHTNESS = 0.06             # effectively black frame
NIGHT_ELEV_DEG = 0.0                # sun below horizon = night


# --------------------------------------------------------------------------- #
# Solar geometry (NOAA algorithm). Vectorized over a DatetimeIndex (UTC).
# --------------------------------------------------------------------------- #
def solar_elevation_deg(lat: float, lon: float, when_utc: pd.DatetimeIndex) -> np.ndarray:
    """Solar elevation angle in degrees for each UTC timestamp at (lat, lon).

    NOAA solar-position equations. lon is east-positive (USGS dec_long_va).
    Accurate to a few hundredths of a degree -- far finer than the day/night cut.
    """
    t = when_utc.tz_convert("UTC") if when_utc.tz is not None else when_utc.tz_localize("UTC")
    doy = t.dayofyear.to_numpy()
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60 + t.second.to_numpy() / 3600
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)

    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    time_offset = eqtime + 4 * lon                 # minutes (timezone=0 for UTC)
    tst = (hour * 60) + time_offset                # true solar time, minutes
    ha = np.radians(tst / 4 - 180)                 # hour angle, radians
    latr = np.radians(lat)
    cos_zen = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(ha)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    return 90.0 - np.degrees(np.arccos(cos_zen))


# --------------------------------------------------------------------------- #
# Photometry (one frame).
# --------------------------------------------------------------------------- #
def _photometry(path: Path) -> dict:
    """Brightness/contrast/blown-out/dark on a 1/4-res gray decode. NaNs on read
    failure so a corrupt frame is dropped, not silently scored as usable."""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return {"brightness": np.nan, "contrast": np.nan,
                "overexposed_frac": np.nan, "dark_frac": np.nan}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "overexposed_frac": float((gray > OVEREXPOSED_LEVEL).mean()),
        "dark_frac": float((gray < DARK_LEVEL).mean()),
    }


def _classify(row: dict) -> dict:
    b, c = row["brightness"], row["contrast"]
    oe = row["overexposed_frac"]
    glare = (not np.isnan(b)) and (oe > GLARE_OVEREXP_FRAC
                                   or (b > GLARE_BRIGHT and c < GLARE_LOW_CONTRAST))
    occluded = (not np.isnan(c)) and (c < OCCLUDED_CONTRAST) and (b > BLACK_BRIGHTNESS)
    black = (not np.isnan(b)) and (b <= BLACK_BRIGHTNESS)
    usable = (not row["is_night"]) and (not glare) and (not occluded) and (not black) \
        and (not np.isnan(b))
    return {"glare": bool(glare), "occluded": bool(occluded), "usable": bool(usable)}


def build_frame_qc(station: str, winter: str, cam_role: str = "primary",
                   workers: int = 8, force: bool = False) -> Path:
    """Build (and cache) the per-frame QC table for one station-winter.

    Reads the camera manifest, filters to the winter window, scores each frame,
    and writes data/interim/<station>/<cam_id>_<winter>_frame_qc.csv. Idempotent:
    skips frames already in the cache unless ``force``."""
    if cv2 is None:
        raise SystemExit("opencv-python (cv2) is required for QC. pip install opencv-python-headless")

    cfg_s = load_yaml("stations.yaml")
    cfg_p = load_yaml("pipeline.yaml")
    d = cfg_s["defaults"]
    st = cfg_s["stations"][station]
    cam_id = st["cameras"][cam_role]["cam_id"]
    lat, lon = st["lat"], st["lon"]

    images_root = Path(cfg_p["paths"]["raw"]) / "images"
    manifest = images_root / station / cam_id / "_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Missing image manifest: {manifest}. Run download_images first.")

    df = pd.read_csv(manifest)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    start, end = winter_bounds(winter, d["season_start_md"], d["season_end_md"])
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["timestamp_utc"] >= lo) & (df["timestamp_utc"] < hi)].copy()
    df = df.sort_values("timestamp_utc").drop_duplicates("filename")
    df = to_hourly(df)   # enforce hourly cadence (collapses burst captures)
    if df.empty:
        raise SystemExit(f"No frames for {station} {winter} in the season window.")

    out_dir = Path(cfg_p["paths"]["interim"]) / station
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cam_id}_{winter}_frame_qc.csv"

    cols = ["filename", "timestamp_utc", "solar_elev_deg", "is_night", "brightness",
            "contrast", "overexposed_frac", "dark_frac", "glare", "occluded", "usable"]

    # Resume: keep only VALIDLY-scored cached rows (brightness not NaN). Frames
    # that failed to decode last time (e.g. mid-download) are NOT treated as done,
    # so they are retried instead of being permanently stuck as unusable.
    prev_valid = pd.DataFrame(columns=cols)
    if out.exists() and not force:
        prev = pd.read_csv(out)
        prev_valid = prev[prev["brightness"].notna()].drop_duplicates("filename", keep="last")
        df = df[~df["filename"].isin(set(prev_valid["filename"]))]
        if df.empty:
            prev_valid.to_csv(out, index=False)   # rewrite clean (drops any stale NaN rows)
            print(f"[qc] {station} {winter}: cache complete ({len(prev_valid)} frames) -> {out}")
            return out

    df["solar_elev_deg"] = solar_elevation_deg(lat, lon, pd.DatetimeIndex(df["timestamp_utc"]))
    df["is_night"] = df["solar_elev_deg"] < NIGHT_ELEV_DEG

    frame_dir = images_root / station / cam_id
    paths = [frame_dir / fn for fn in df["filename"]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        photo = list(ex.map(_photometry, paths))
    photo = pd.DataFrame(photo, index=df.index)
    df = pd.concat([df, photo], axis=1)

    cls = pd.DataFrame([_classify(r) for r in df.to_dict("records")], index=df.index)
    df = pd.concat([df, cls], axis=1)

    # Merge freshly-scored rows with valid cached rows and rewrite the whole file,
    # so the cache is always duplicate-free and carries no stale NaN rows.
    full = df[cols].copy() if prev_valid.empty else pd.concat([prev_valid, df[cols]], ignore_index=True)
    full["timestamp_utc"] = pd.to_datetime(full["timestamp_utc"], utc=True)
    full = full.drop_duplicates("filename", keep="last").sort_values("timestamp_utc")
    full.to_csv(out, index=False)

    print(f"[qc] {station} {winter}: scored {len(df)} frames "
          f"({len(prev_valid)} reused) -> {out}")
    print(f"     night {full.is_night.mean():.0%}  glare {full.glare.mean():.0%}  "
          f"occluded {full.occluded.mean():.0%}  usable {full.usable.mean():.0%}")
    return out


def load_frame_qc(station: str, winter: str, cam_id: str, interim_root: Path) -> pd.DataFrame | None:
    p = interim_root / station / f"{cam_id}_{winter}_frame_qc.csv"
    if not p.exists():
        return None
    q = pd.read_csv(p)
    q["timestamp_utc"] = pd.to_datetime(q["timestamp_utc"], utc=True)
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--winter", required=True)
    ap.add_argument("--cam-role", default="primary")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rescore even if cached")
    args = ap.parse_args()
    build_frame_qc(args.station, args.winter, args.cam_role, args.workers, args.force)


if __name__ == "__main__":
    main()
