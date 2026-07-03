"""Assemble hourly images into fixed-length clips and align exogenous series.

This is the data-plumbing step the research plan calls the hidden time sink.
It turns per-image rows into the clip records the encoder and temporal head consume.

Steps:
  1. Read data/raw/images/<station>/<camId>/_manifest.csv (filename, timestamp_utc).
  2. Resample/group into clips of `clip.frames` consecutive hourly frames,
     hopping by `clip.stride_hours`. Drop clips with < `min_valid_frames`.
  3. Join mean air temperature (from download_temperature output) per clip.
  4. Join stage statistics (from download_stage output) per clip.
  5. Attach the per-clip ice label (fm_ice.data.labels) and per-clip QC
     aggregates (fm_ice.data.qc).
  6. Write a clip manifest: data/interim/<station>/clips_<winter>.parquet
       columns: clip_id, station, cam_id, t_start_utc, t_end_utc, n_frames,
                max_gap_hours, frame_paths (list), air_tmpc_mean, stage_mean,
                ice_flag, ice_day_frac, ice_explicit, ice_estimated,
                n_usable, valid_fraction, night_frac, glare_frac, occluded_frac,
                brightness_mean

The per-clip ice label comes from the USGS daily ice flag (fm_ice.data.labels).
Stage breakpoints are an independent onset/breakup reference and live in
fm_ice.evaluation.reference_events, not here.

Usage:
  python -m fm_ice.data.assemble_clips --station cedarburg --winter 2024-2025
  python -m fm_ice.data.assemble_clips --station cedarburg --winter 2024-2025 --no-qc

Status: WORKING -- grouping, joins, labels, and QC are all implemented.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from fm_ice.config import load_yaml, station_utc_offset_hours, winter_bounds
from fm_ice.data.labels import label_clips, load_ice_flags
from fm_ice.data import qc as qcmod


def load_image_manifest(station: str, cam_id: str, images_root: Path) -> pd.DataFrame:
    m = images_root / station / cam_id / "_manifest.csv"
    if not m.exists():
        raise SystemExit(f"Missing image manifest: {m}. Run download_images first.")
    df = pd.read_csv(m)
    # Normalize to tz-aware UTC so all downstream joins compare cleanly.
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")


def filter_to_winter(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Keep only frames inside this winter's season window. The per-camera
    manifest accumulates every winter ever downloaded, so a winter-specific run
    must slice it; otherwise clips span multiple winters."""
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)  # inclusive of end day
    return df[(df["timestamp_utc"] >= lo) & (df["timestamp_utc"] < hi)]


def group_clips(df: pd.DataFrame, frames: int, stride_hours: int, min_valid: int) -> list[dict]:
    """Group hourly rows into clips. Each clip is `frames` hours wide, hopping
    by `stride_hours`. Robust to gaps: a clip keeps whatever real frames fall in
    its window and is dropped if fewer than `min_valid`."""
    if df.empty:
        return []
    t0 = df["timestamp_utc"].min().floor("h")
    t_end = df["timestamp_utc"].max().ceil("h")
    clips = []
    start = t0
    cid = 0
    while start < t_end:
        stop = start + pd.Timedelta(hours=frames)
        win = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] < stop)]
        if len(win) >= min_valid:
            ts = win["timestamp_utc"].sort_values()
            # Largest gap between consecutive real frames inside the clip, plus the
            # gaps at the clip edges (start->first, last->stop). Hourly cadence
            # means a clean clip has max_gap_hours ~= 1.
            edges = pd.Series([start, *ts.tolist(), stop])
            max_gap = edges.diff().dropna().max()
            clips.append({
                "clip_id": cid,
                "t_start_utc": start,
                "t_end_utc": stop,
                "n_frames": len(win),
                "max_gap_hours": float(max_gap.total_seconds() / 3600.0),
                "frame_paths": list(win["filename"]),
            })
            cid += 1
        start = start + pd.Timedelta(hours=stride_hours)
    return clips


def join_qc(clips: pd.DataFrame, qc: pd.DataFrame | None, frames: int) -> pd.DataFrame:
    """Aggregate per-frame QC to per-clip: usable fraction and bad-frame rates.

    valid_fraction is usable frames over the EXPECTED frame count (clip width),
    so both missing frames and unusable frames pull it down. If no QC table is
    available the columns are pd.NA (unknown), not 0."""
    if qc is None or qc.empty:
        for col in ("n_usable", "valid_fraction", "night_frac", "glare_frac",
                    "occluded_frac", "brightness_mean"):
            clips[col] = pd.NA
        return clips

    # Drop any duplicate filenames so a corrupted/concatenated cache cannot abort
    # the run with a reindex-on-duplicate-labels error.
    q = qc.drop_duplicates("filename", keep="first").set_index("filename")
    if len(q) < len(qc):
        print(f"[assemble] join_qc: dropped {len(qc) - len(q)} duplicate QC rows")
    n_usable, valid_frac, night, glare, occl, bright = [], [], [], [], [], []
    for _, c in clips.iterrows():
        sub = q.reindex([f for f in c["frame_paths"] if f in q.index])
        n = len(sub)
        n_use = int(sub["usable"].sum()) if n else 0
        # No QC coverage for this clip -> unknown (NA), not "0 usable" (which would
        # read as a fully-invalid clip).
        n_usable.append(n_use if n else pd.NA)
        valid_frac.append(n_use / frames if n else pd.NA)
        night.append(float(sub["is_night"].mean()) if n else pd.NA)
        glare.append(float(sub["glare"].mean()) if n else pd.NA)
        occl.append(float(sub["occluded"].mean()) if n else pd.NA)
        bright.append(float(sub["brightness"].mean()) if n else pd.NA)
    clips["n_usable"] = n_usable
    clips["valid_fraction"] = valid_frac
    clips["night_frac"] = night
    clips["glare_frac"] = glare
    clips["occluded_frac"] = occl
    clips["brightness_mean"] = bright
    return clips


def join_temperature(clips: pd.DataFrame, temp_csv: Path) -> pd.DataFrame:
    if not temp_csv.exists():
        clips["air_tmpc_mean"] = pd.NA
        return clips
    t = pd.read_csv(temp_csv)
    t["datetime_utc"] = pd.to_datetime(t["datetime_utc"], utc=True)
    t = t.sort_values("datetime_utc")
    means = []
    for _, c in clips.iterrows():
        m = t[(t["datetime_utc"] >= c["t_start_utc"]) & (t["datetime_utc"] < c["t_end_utc"])]["tmpc"]
        means.append(m.mean() if len(m) else pd.NA)
    clips["air_tmpc_mean"] = means
    return clips


def join_stage(clips: pd.DataFrame, stage_csv: Path) -> pd.DataFrame:
    if not stage_csv.exists():
        clips["stage_mean"] = pd.NA
        return clips
    s = pd.read_csv(stage_csv)
    s["datetime"] = pd.to_datetime(s["datetime"], utc=True)
    # Keep only gage height (00065); discharge (00060) shares the file.
    s = s[s["param"].astype(str).str.contains("65")] if "param" in s else s
    s = s.sort_values("datetime")
    means = []
    for _, c in clips.iterrows():
        m = s[(s["datetime"] >= c["t_start_utc"]) & (s["datetime"] < c["t_end_utc"])]["value"]
        v = pd.to_numeric(m, errors="coerce")
        # USGS NWIS uses numeric -999999 sentinels for missing/ice-affected stage
        # (common in exactly the onset/breakup windows); mask before averaging.
        v = v[v > -999990]
        means.append(v.mean() if len(v) else pd.NA)
    clips["stage_mean"] = means
    return clips


def run(station: str, winter: str, cam_role: str, do_qc: bool = True) -> None:
    cfg_s = load_yaml("stations.yaml")
    cfg_p = load_yaml("pipeline.yaml")
    paths = cfg_p["paths"]
    clip = cfg_p["clip"]
    defaults = cfg_s["defaults"]
    st = cfg_s["stations"][station]
    cam_id = st["cameras"][cam_role]["cam_id"]
    nwis = st["nwis_id"]
    asos = st["asos_station"]

    images_root = Path(paths["raw"]) / "images"
    df = load_image_manifest(station, cam_id, images_root)

    start, end = winter_bounds(winter, cfg_s["defaults"]["season_start_md"],
                               cfg_s["defaults"]["season_end_md"])
    df = filter_to_winter(df, start, end)
    df = qcmod.to_hourly(df)   # enforce hourly cadence (matches the QC frame set)
    if df.empty:
        print(f"[assemble] {station} {winter}: no frames in season window "
              f"({start}..{end}). Did download_images run for this winter?")
        return

    clips = group_clips(df, clip["frames"], clip["stride_hours"], clip["min_valid_frames"])
    clips = pd.DataFrame(clips)
    if clips.empty:
        print(f"[assemble] {station} {winter}: no clips formed. Check cadence/coverage.")
        return
    clips["station"] = station
    clips["cam_id"] = cam_id

    temp_csv = Path(paths["raw"]) / "temperature" / station / f"{asos}_{winter}_tmp.csv"
    stage_csv = Path(paths["raw"]) / "stage" / station / f"{nwis}_{winter}_iv.csv"
    clips = join_temperature(clips, temp_csv)
    clips = join_stage(clips, stage_csv)

    # Labels: per-clip ice-affected state from the USGS daily ice flag, mapped
    # through the local-standard-time day boundary. (Stage-breakpoint events are
    # an independent reference in fm_ice.evaluation.reference_events.)
    ice = load_ice_flags(nwis, winter, station, Path(paths["raw"]) / "ice_flags")
    if ice is None:
        print(f"[assemble] {station} {winter}: no ice-flag CSV; labels = NA. "
              f"Run download_ice_flags to populate labels.")
    clips = label_clips(clips, ice, station_utc_offset_hours(station))

    # QC: night/glare/occlusion per frame -> usable fraction per clip.
    qc = None
    if do_qc:
        qcmod.build_frame_qc(station, winter, cam_role)   # idempotent (cached)
        qc = qcmod.load_frame_qc(station, winter, cam_id, Path(paths["interim"]))
    else:
        qc = qcmod.load_frame_qc(station, winter, cam_id, Path(paths["interim"]))
        if qc is None:
            print(f"[assemble] {station} {winter}: --no-qc and no cached QC; "
                  f"QC columns = NA.")
    clips = join_qc(clips, qc, clip["frames"])

    out_dir = Path(paths["interim"]) / station
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"clips_{winter}.parquet"
    clips.to_parquet(out, index=False)
    n_ice = int(pd.to_numeric(clips["ice_flag"], errors="coerce").fillna(0).sum())
    vf = pd.to_numeric(clips["valid_fraction"], errors="coerce")
    vf_txt = f", mean valid_fraction {vf.mean():.2f}" if vf.notna().any() else ""
    print(f"[assemble] {station} {winter}: {len(clips)} clips -> {out}")
    print(f"           ice-labeled clips: {n_ice}/{len(clips)} "
          f"({n_ice / len(clips):.0%}){vf_txt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--winter", required=True)
    ap.add_argument("--cam-role", default="primary")
    ap.add_argument("--no-qc", action="store_true",
                    help="skip the image-QC pass (use cached QC if present, else NA)")
    args = ap.parse_args()
    run(args.station, args.winter, args.cam_role, do_qc=not args.no_qc)


if __name__ == "__main__":
    main()
