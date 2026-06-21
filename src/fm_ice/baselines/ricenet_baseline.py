"""RIce-Net threshold baseline: the anchor this project must beat.

Pipeline (Ayyad et al. 2025), reproduced as the baseline:
  1. Per image: segment ice vs water over the river mask, then
  2. Ice coverage IC(t) = ice_pixels / river_pixels * 100.
  3. Flag rule: flag on at t if IC(t') > tau for all t' in [t-8h, t].
       tau = 15 for onset, tau = 20 for breakup.
  4. Onset = first flag-on transition; breakup = corresponding flag-off in spring.

Steps 1-2 (segment_ice_coverage) are implemented here against the published
RIce-Net release, verified 2026-06-20:
  * Code:    github.com/m-ayyad/RIce-Net  (MIT). The saved .pth is a PICKLED FULL
             MODEL object -- load with torch.load(weights_only=False) and call
             .forward() directly; segmentation_models_pytorch must be importable
             at unpickle time. Architecture (paper: PAN/ResNet50) is not needed to
             run, only to rebuild from a state_dict if the pickle ever breaks.
  * Weights: HydroShare DOI 10.4211/hs.ff4e9c4e87ef4d7d923efe77f5ed2b83 (CC BY
             4.0). segmentation_model.pth ~93 MB. See download_weights().
  * Mask:    NOT a file -- the per-site river polygon/crop/pixels live in
             configs/stations.yaml under cedarburg.cameras.primary.ricenet_mask
             (copied from their config.py). Only Cedarburg ships a mask; Bismarck
             would need a hand-digitized polygon.
  * Preproc: mask (fillPoly+bitwise_and) -> crop -> resize to 1152x640 INTER_CUBIC
             -> BGR2RGB -> /255 -> (x-mean)/std (ImageNet) -> CHW. ice = class 1.

torch / cv2 / smp are intentionally NOT hard imports (the encoder is meant to be
the only heavy dependency); they are imported lazily inside the functions that
need them, so the threshold-rule code below stays testable on a plain CPU.

Output: results/ricenet_events.csv and results/ricenet_coverage_<station>_<winter>.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml, winter_bounds

ONSET_TAU = 15.0
BREAKUP_TAU = 20.0
PERSIST_HOURS = 8


# --------------------------------------------------------------------------- #
# Segmentation (the part that needs the published model + mask).
# --------------------------------------------------------------------------- #
def build_river_mask(polygon: list, shape: tuple[int, int]) -> "np.ndarray":
    """Filled binary stencil (H, W) from the river polygon in image pixel coords."""
    import cv2
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask


def _preprocess(path: Path, mask_cfg: dict, rc: dict) -> "np.ndarray":
    """Replicate the RIce-Net per-image transform. Returns a (1,3,H,W) float32
    array ready for the model, or None if the image cannot be read."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)   # BGR, full 1920x1080
    if img is None:
        return None
    mask = build_river_mask(mask_cfg["polygon"], img.shape[:2])
    img = cv2.bitwise_and(img, img, mask=mask)       # zero outside the river
    y0, y1, x0, x1 = mask_cfg["crop"]
    img = img[y0:y1, x0:x1, :]
    img = cv2.resize(img, (rc["input_width"], rc["input_height"]), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # unconditional, per recipe
    img = (img - np.array(rc["mean"], np.float32)) / np.array(rc["std"], np.float32)
    return np.transpose(img, (2, 0, 1))[None, ...]   # (1,3,H,W)


def load_segmentation_model(weights_path: Path):
    """Unpickle the full RIce-Net segmentation model onto CPU. Requires torch and
    (for unpickling) segmentation_models_pytorch importable."""
    import torch
    try:
        import segmentation_models_pytorch  # noqa: F401  (needed at unpickle time)
    except ImportError as e:
        raise SystemExit("segmentation_models_pytorch must be installed to unpickle "
                          "the RIce-Net weights. pip install segmentation-models-pytorch") from e
    if not Path(weights_path).exists():
        raise SystemExit(f"Missing RIce-Net weights: {weights_path}. Run "
                         f"`python -m fm_ice.baselines.ricenet_baseline --download`.")
    model = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.eval()
    return model


def segment_ice_coverage(image_paths, timestamps, mask_cfg: dict, rc: dict, model) -> pd.Series:
    """Run the segmenter over each image, restrict to the river mask, and return
    IC(t) in percent indexed by timestamp. IC = ice_pixels / pixels * 100, where
    `pixels` is the RIce-Net river-surface denominator for this site.

    This is the ONLY place the segmentation model is used, so the rest of the
    baseline stays model-agnostic and testable on synthetic coverage series."""
    import torch
    ice_idx = rc["ice_class_index"]
    denom = float(mask_cfg["pixels"])
    cov, idx = [], []
    with torch.no_grad():
        for p, t in zip(image_paths, timestamps):
            x = _preprocess(Path(p), mask_cfg, rc)
            if x is None:
                continue
            logits = model.forward(torch.from_numpy(x))
            pred = np.argmax(logits.squeeze(0).cpu().numpy(), axis=0)   # (H,W)
            cov.append(float((pred == ice_idx).sum()) / denom * 100.0)
            idx.append(t)
    return pd.Series(cov, index=pd.DatetimeIndex(idx), name="ice_coverage")


# --------------------------------------------------------------------------- #
# Threshold-persistence rule (model-agnostic, exact, unit-tested).
# --------------------------------------------------------------------------- #
def threshold_persistence_flag(coverage: pd.Series, tau: float, persist_hours: int = PERSIST_HOURS) -> pd.Series:
    """Flag is on at t if coverage > tau for the whole trailing persist_hours window.

    coverage: float Series indexed by tz-aware UTC datetimes (hourly or finer).
    Returns a boolean Series aligned to coverage.index.
    """
    cov = coverage.sort_index()
    above = cov > tau
    flags = []
    persist = pd.Timedelta(hours=persist_hours)
    for t in cov.index:
        window = above.loc[t - persist:t]
        # Require a FULL trailing window, all above tau: the window must reach back
        # a real persist_hours (span check) AND have enough samples (>= persist_hours,
        # tolerating ~1 missing hourly frame). Without this the flag fires at the
        # series start or across a large gap, producing a spurious onset/breakup.
        spans = (t - window.index.min()) >= persist
        flags.append(spans and len(window) >= persist_hours and bool(window.all()))
    return pd.Series(flags, index=cov.index)


def onset_breakup_dates(coverage: pd.Series, onset_tau: float = ONSET_TAU,
                        breakup_tau: float = BREAKUP_TAU,
                        persist_hours: int = PERSIST_HOURS) -> dict[str, object]:
    """Read onset and breakup from the coverage series using the RIce-Net rule.

    Onset: first rising edge of the onset-threshold flag.
    Breakup: first falling edge of the breakup-threshold flag after onset.
    """
    onset_flag = threshold_persistence_flag(coverage, onset_tau, persist_hours)
    breakup_flag = threshold_persistence_flag(coverage, breakup_tau, persist_hours)

    onset = None
    rises = onset_flag & ~onset_flag.shift(1, fill_value=False)
    if rises.any():
        onset = rises[rises].index[0]

    breakup = None
    if onset is not None:
        after = breakup_flag.loc[onset:]
        falls = ~after & after.shift(1, fill_value=False)
        if falls.any():
            breakup = falls[falls].index[0]
    return {"onset": onset, "breakup": breakup}


# --------------------------------------------------------------------------- #
# Weights download + end-to-end run.
# --------------------------------------------------------------------------- #
def download_weights(rc: dict) -> None:
    """Fetch the RIce-Net weights from HydroShare into rc['weights_dir'].

    Uses hsclient (selective, ~93 MB) if available; otherwise prints the BagIt
    fallback. Public resource, CC BY 4.0 -- no login needed for the bag."""
    out = Path(rc["weights_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rid = rc["hydroshare_id"]
    try:
        from hsclient import HydroShare
        hs = HydroShare()
        res = hs.resource(rid)
        for fn in (rc["segmentation_weights"], rc["classification_weights"]):
            print(f"[ricenet] downloading {fn} ...")
            res.file_download(fn, save_path=str(out))
        print(f"[ricenet] weights -> {out}")
    except ImportError:
        print("hsclient not installed. Either `pip install hsclient` and re-run "
              "--download, or fetch the BagIt zip manually (CC BY 4.0):\n"
              f"  https://www.hydroshare.org/django_irods/rest_download/bags/{rid}.zip\n"
              f"  then unzip and copy *.pth into {out}/")


def _winter_frames(station: str, winter: str, cam_role: str = "primary") -> pd.DataFrame:
    """Hourly frame manifest (filename, timestamp_utc, path) for a station-winter."""
    from fm_ice.data.qc import to_hourly
    cfg_s = load_yaml("stations.yaml")
    cfg_p = load_yaml("pipeline.yaml")
    d = cfg_s["defaults"]
    cam_id = cfg_s["stations"][station]["cameras"][cam_role]["cam_id"]
    root = Path(cfg_p["paths"]["raw"]) / "images" / station / cam_id
    df = pd.read_csv(root / "_manifest.csv")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    start, end = winter_bounds(winter, d["season_start_md"], d["season_end_md"])
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["timestamp_utc"] >= lo) & (df["timestamp_utc"] < hi)]
    df = to_hourly(df.sort_values("timestamp_utc").drop_duplicates("filename"))
    df["path"] = [str(root / fn) for fn in df["filename"]]
    return df


def run(station: str, winter: str, cam_role: str = "primary") -> dict:
    """Full RIce-Net baseline for one station-winter: segment every hourly frame,
    compute IC(t), apply the threshold rule, and persist coverage + events."""
    cfg_s = load_yaml("stations.yaml")
    cfg_p = load_yaml("pipeline.yaml")
    rc = cfg_p["ricenet"]
    if rc.get("use_classifier_gate"):
        raise SystemExit("ricenet.use_classifier_gate=true but the classifier gate "
                         "is not wired in segment_ice_coverage. Wire it (load "
                         "classification_model.pth and zero IC on 'no ice') or set "
                         "use_classifier_gate: false in configs/pipeline.yaml.")
    cam = cfg_s["stations"][station]["cameras"][cam_role]
    if "ricenet_mask" not in cam:
        raise SystemExit(f"No RIce-Net river mask for {station}/{cam_role}. Only "
                         f"sites with a digitized polygon in stations.yaml can run "
                         f"this baseline (Cedarburg ships one; Bismarck does not).")
    mask_cfg = cam["ricenet_mask"]

    frames = _winter_frames(station, winter, cam_role)
    print(f"[ricenet] {station} {winter}: segmenting {len(frames)} hourly frames "
          f"(CPU is fine but slow) ...")
    model = load_segmentation_model(Path(rc["weights_dir"]) / rc["segmentation_weights"])
    cov = segment_ice_coverage(frames["path"], frames["timestamp_utc"], mask_cfg, rc, model)

    events = onset_breakup_dates(cov, rc["onset_tau"], rc["breakup_tau"], rc["persist_hours"])

    results = Path(cfg_p["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    cov.rename("ice_coverage").to_frame().to_csv(results / f"ricenet_coverage_{station}_{winter}.csv")
    print(f"[ricenet] {station} {winter}: onset={events['onset']}  breakup={events['breakup']}")
    return {"station": station, "winter": winter, **events}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station")
    ap.add_argument("--winter")
    ap.add_argument("--cam-role", default="primary")
    ap.add_argument("--download", action="store_true", help="fetch RIce-Net weights from HydroShare")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    cfg_p = load_yaml("pipeline.yaml")
    if args.download:
        download_weights(cfg_p["ricenet"])
        return
    if not (args.station and args.winter):
        raise SystemExit("Provide --station and --winter, or --download.")

    ev = run(args.station, args.winter, args.cam_role)
    out = Path(args.results) / "ricenet_events.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["station", "winter", "onset_utc", "breakup_utc"])
        w.writerow([ev["station"], ev["winter"], ev["onset"], ev["breakup"]])
    print(f"[ricenet] appended event row -> {out}")


if __name__ == "__main__":
    # Self-test on a synthetic coverage series: ramps up in Jan, down in Mar.
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        idx = pd.date_range("2025-12-01", "2026-04-01", freq="h", tz="UTC")
        cov = pd.Series(np.zeros(len(idx)), index=idx)
        cov.loc["2026-01-10":"2026-03-05"] = 60.0
        print(onset_breakup_dates(cov))
