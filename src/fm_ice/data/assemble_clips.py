"""Assemble hourly images into fixed-length clips and align exogenous series.

This is the data-plumbing step the research plan calls the hidden time sink.
It turns per-image rows into the clip records the encoder and temporal head consume.

Steps:
  1. Read data/raw/images/<station>/<camId>/_manifest.csv (filename, timestamp_utc).
  2. Resample/group into clips of `clip.frames` consecutive hourly frames,
     hopping by `clip.stride_hours`. Drop clips with < `min_valid_frames`.
  3. Join mean air temperature (from download_temperature output) per clip.
  4. Join stage statistics (from download_stage output) per clip.
  5. Write a clip manifest: data/interim/<station>/clips_<winter>.parquet
       columns: clip_id, station, cam_id, t_start_utc, t_end_utc,
                frame_paths (list), n_frames, air_tmpc_mean, stage_mean, label(optional)

Labels (ice-affected state per clip) are attached separately from the USGS ice
flag and stage breakpoints; see TODO below and fm_ice.baselines.

Usage:
  python -m fm_ice.data.assemble_clips --station cedarburg --winter 2024-2025

Status: WORKING for grouping + joins. Labeling and QC flags are marked TODO.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fm_ice.config import load_yaml


def load_image_manifest(station: str, cam_id: str, images_root: Path) -> pd.DataFrame:
    m = images_root / station / cam_id / "_manifest.csv"
    if not m.exists():
        raise SystemExit(f"Missing image manifest: {m}. Run download_images first.")
    df = pd.read_csv(m, parse_dates=["timestamp_utc"])
    return df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")


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
            clips.append({
                "clip_id": cid,
                "t_start_utc": start,
                "t_end_utc": stop,
                "n_frames": len(win),
                "frame_paths": list(win["filename"]),
            })
            cid += 1
        start = start + pd.Timedelta(hours=stride_hours)
    return clips


def join_temperature(clips: pd.DataFrame, temp_csv: Path) -> pd.DataFrame:
    if not temp_csv.exists():
        clips["air_tmpc_mean"] = pd.NA
        return clips
    t = pd.read_csv(temp_csv, parse_dates=["datetime_utc"]).sort_values("datetime_utc")
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
    s = pd.read_csv(stage_csv, parse_dates=["datetime"])
    s = s[s["param"] == 65] if "param" in s and s["param"].dtype != object else s
    s = s[s["param"].astype(str).str.contains("65")] if "param" in s else s
    s = s.sort_values("datetime")
    means = []
    for _, c in clips.iterrows():
        m = s[(s["datetime"] >= c["t_start_utc"]) & (s["datetime"] < c["t_end_utc"])]["value"]
        means.append(pd.to_numeric(m, errors="coerce").mean() if len(m) else pd.NA)
    clips["stage_mean"] = means
    return clips


def run(station: str, winter: str, cam_role: str) -> None:
    cfg_s = load_yaml("stations.yaml")
    cfg_p = load_yaml("pipeline.yaml")
    paths = cfg_p["paths"]
    clip = cfg_p["clip"]
    st = cfg_s["stations"][station]
    cam_id = st["cameras"][cam_role]["cam_id"]
    nwis = st["nwis_id"]
    asos = st["asos_station"]

    images_root = Path(paths["raw"]) / "images"
    df = load_image_manifest(station, cam_id, images_root)

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

    # TODO(label): attach ice-affected label per clip from ice_flags + stage breakpoints.
    # TODO(qc): flag night/glare/occluded frames; record valid_fraction per clip.

    out_dir = Path(paths["interim"]) / station
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"clips_{winter}.parquet"
    clips.to_parquet(out, index=False)
    print(f"[assemble] {station} {winter}: {len(clips)} clips -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--winter", required=True)
    ap.add_argument("--cam-role", default="primary")
    args = ap.parse_args()
    run(args.station, args.winter, args.cam_role)


if __name__ == "__main__":
    main()
