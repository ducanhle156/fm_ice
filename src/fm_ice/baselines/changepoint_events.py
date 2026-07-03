"""Label-free onset/breakup events from the per-clip EMBEDDING stream.

This is the Phase-4 change-point baseline that the H2 table scores alongside the
temporal head: it asks how well ice onset/breakup can be dated WITHOUT any labels,
purely from how the frozen FM embeddings drift over the winter. It mirrors the
ranking-by-level-shift logic of evaluation.reference_events.stage_breakpoint_events,
but the signal is the embedding sequence rather than the stage gauge.

Pipeline (all CPU, reads the cached .npy embeddings + row index):
  1. Load embeddings for a station-winter and order rows by t_start_utc (UTC).
  2. Reduce the (T, 1024) sequence to a 1-D signal:
       pc1       first principal component of the z-scored embedding sequence
                 (overall appearance drift; ice flips the dominant axis).
       diffnorm  L2 norm of consecutive embedding differences (a motion /
                 appearance-shift proxy; spikes where the scene changes regime).
     The 1-D signal is standardized before change detection.
  3. Detect change points:
       bocpd  fm_ice.baselines.changepoint.bocpd_gaussian +
              changepoints_from_runlength. hazard_lambda in the config is in
              HOURS; clips are clip.stride_hours apart, so the expected segment
              length in *steps* is hazard_lambda / stride_hours.
       beast  fm_ice.baselines.changepoint.beast_changepoints, then threshold the
              trend change-point probability (.trend.cpOccPr).
  4. Map change points to onset (dominant change in local-time months Nov-Feb) and
     breakup (dominant in Mar-Apr), ranked by local level-shift magnitude on the
     1-D signal -- the same ranking idea as stage_breakpoint_events. Local-month
     bucketing uses defaults.usgs_dv_utc_offset_hours.

Output: results/changepoint_events.csv
  station, winter, encoder, method, signal, onset_utc, breakup_utc,
  n_changepoints, notes

Usage:
  python -m fm_ice.baselines.changepoint_events --encoder vjepa2 --station cedarburg --winter 2022-2023
  python -m fm_ice.baselines.changepoint_events --encoder dinov2 --all --method bocpd --signal pc1
  python -m fm_ice.baselines.changepoint_events --all --method beast --signal diffnorm
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml
from fm_ice.baselines.changepoint import (
    bocpd_gaussian,
    changepoints_from_runlength,
    beast_changepoints,
)
from fm_ice.evaluation.probe_separability import load_probe_matrix

# The two cedarburg train winters (bismarck 2024-2025 is the Phase-5 transfer test
# and is deliberately NOT touched in Phase 4).
PAPER_TRAIN_WINTERS = [("cedarburg", "2022-2023"), ("cedarburg", "2023-2024")]

# Default BEAST trend change-point probability threshold, exposed as --beast-thresh.
# BEAST's per-timepoint cpOccPr is diffuse for a long (~1100-step) winter series --
# the posterior change mass spreads over neighbouring steps, so the max cpOccPr is
# only ~0.18 here and an absolute 0.5 never fires. 0.1 matches the observed scale.
BEAST_CP_PROB_THRESH = 0.1


# --------------------------------------------------------------------------- #
# Load the ordered embedding sequence for a station-winter.
# --------------------------------------------------------------------------- #
def load_embedding_series(encoder: str, station: str, winter: str,
                          stem: str | None = None):
    """Return (X, times) for ALL clips of a station-winter, ordered by t_start_utc.

    X is (T, D) float32; times is a tz-aware UTC DatetimeIndex aligned to X. Unlike
    load_probe_matrix this keeps every clip (we want the full sequence, labelled or
    not), so we re-load the raw cache rather than the label-filtered probe matrix.
    """
    cfg = load_yaml("pipeline.yaml")
    cache = Path(cfg["paths"]["cache"]) / encoder / station
    stem = stem or winter
    npy = cache / f"{stem}.npy"
    idx_csv = cache / f"{stem}_index.csv"
    if not npy.exists():
        raise SystemExit(f"missing embeddings {npy} -- run extract_embeddings first")

    X = np.load(npy).astype("float32")
    index = pd.read_csv(idx_csv)
    if len(index) != len(X):
        raise SystemExit(f"row mismatch: {len(X)} embeddings vs {len(index)} index rows")
    times = pd.to_datetime(index["t_start_utc"], utc=True)
    order = np.argsort(times.to_numpy(), kind="stable")
    return X[order], times.iloc[order].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Reduce (T, D) -> 1-D signal.
# --------------------------------------------------------------------------- #
def reduce_signal(X: np.ndarray, signal: str) -> np.ndarray:
    """Project the embedding sequence to a 1-D series for change detection.

    pc1      first principal component of the z-scored embeddings.
    diffnorm L2 norm of consecutive embedding differences (length T, with a
             leading 0 so indices stay aligned to the clip times).
    The returned signal is standardized (zero mean, unit std).
    """
    if signal == "pc1":
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-9
        Z = (X - mu) / sd
        # First PC via SVD on the centered (already z-scored) matrix.
        Zc = Z - Z.mean(axis=0)
        # economy SVD; right-singular vector 0 is the PC1 loading.
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        sig = Zc @ Vt[0]
    elif signal == "diffnorm":
        d = np.linalg.norm(np.diff(X, axis=0), axis=1)
        sig = np.concatenate([[0.0], d])
    else:
        raise ValueError(f"unknown signal {signal!r} (use pc1 or diffnorm)")
    sig = np.asarray(sig, dtype=float)
    return (sig - sig.mean()) / (sig.std() + 1e-9)


# --------------------------------------------------------------------------- #
# Detect change points on the 1-D signal.
# --------------------------------------------------------------------------- #
def detect_changepoints(sig: np.ndarray, method: str, *, hazard_steps: float,
                        min_drop: int, min_sep: int, season_period: int,
                        beast_thresh: float) -> list[int]:
    """Return change-point indices into `sig`.

    bocpd via bocpd_gaussian + changepoints_from_runlength (hazard in *steps*).
    beast via beast_changepoints, thresholding the trend cpOccPr.
    """
    if method == "bocpd":
        res = bocpd_gaussian(sig, hazard_lambda=hazard_steps)
        return changepoints_from_runlength(res["map_run_length"],
                                           min_drop=min_drop, min_sep=min_sep)
    if method == "beast":
        o = beast_changepoints(sig, season_period=season_period)
        prob = np.asarray(o.trend.cpOccPr, dtype=float)
        cps = [int(i) for i in np.where(prob >= beast_thresh)[0]]
        # enforce a minimum separation, keeping the higher-probability point.
        cps.sort(key=lambda i: -prob[i])
        kept: list[int] = []
        for i in cps:
            if all(abs(i - j) >= min_sep for j in kept):
                kept.append(i)
        return sorted(kept)
    raise ValueError(f"unknown method {method!r} (use bocpd or beast)")


# --------------------------------------------------------------------------- #
# Map change points -> onset / breakup (mirrors stage_breakpoint_events).
# --------------------------------------------------------------------------- #
def _level_shift(sig: np.ndarray, i: int, half: int) -> float:
    """Magnitude of the level shift in `sig` at index i (mean after vs before)."""
    a = sig[max(0, i - half):i]
    b = sig[i:i + half]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return abs(float(b.mean()) - float(a.mean()))


def pick_events(sig: np.ndarray, times: pd.Series, cps: list[int],
                offset_hours: int, half: int = 12) -> dict:
    """Onset = dominant change point (largest level shift) in local months Nov-Feb;
    breakup = dominant in Mar-Apr. Local month = UTC time + offset_hours (CST)."""
    local_month = (times + pd.Timedelta(hours=-offset_hours)).dt.month

    onset_cands = [(_level_shift(sig, i, half), times.iloc[i])
                   for i in cps if local_month.iloc[i] in (11, 12, 1, 2)]
    breakup_cands = [(_level_shift(sig, i, half), times.iloc[i])
                     for i in cps if local_month.iloc[i] in (3, 4)]
    onset = max(onset_cands)[1] if onset_cands else None
    breakup = max(breakup_cands)[1] if breakup_cands else None
    return {"onset": onset, "breakup": breakup}


# --------------------------------------------------------------------------- #
def run_one(encoder: str, station: str, winter: str, method: str, signal: str,
            stem: str | None = None,
            beast_thresh: float = BEAST_CP_PROB_THRESH) -> dict:
    cfg = load_yaml("pipeline.yaml")
    cfg_s = load_yaml("stations.yaml")
    offset = int(cfg_s["stations"][station].get(
        "usgs_dv_utc_offset_hours", cfg_s["defaults"]["usgs_dv_utc_offset_hours"]))
    stride = cfg["clip"]["stride_hours"]
    hazard_hours = cfg["changepoint"]["bocpd"]["hazard_lambda"]
    season_period = cfg["changepoint"]["beast"]["season_period"]
    # config hazard is in hours; convert to expected segment length in steps.
    hazard_steps = float(hazard_hours) / float(stride)
    # min_drop / min_sep scaled to steps: a real regime change should persist at
    # least ~2 days, and onset/breakup are weeks apart.
    min_drop = max(1, int(round(24 / stride)))     # ~1 day of run-length reset
    min_sep = max(1, int(round(48 / stride)))      # >= ~2 days between events

    X, times = load_embedding_series(encoder, station, winter, stem=stem)
    sig = reduce_signal(X, signal)
    cps = detect_changepoints(sig, method, hazard_steps=hazard_steps,
                              min_drop=min_drop, min_sep=min_sep,
                              season_period=season_period,
                              beast_thresh=beast_thresh)
    ev = pick_events(sig, times, cps, offset)

    row = {
        "station": station, "winter": winter, "encoder": encoder,
        "method": method, "signal": signal,
        "onset_utc": ev["onset"], "breakup_utc": ev["breakup"],
        "n_changepoints": len(cps),
        "notes": (f"hazard={hazard_hours}h/{stride}h={hazard_steps:.0f}steps"
                  if method == "bocpd" else f"beast_thresh={beast_thresh}"),
    }
    print(f"[cp] {station} {winter} {encoder} {method}/{signal}: "
          f"onset={row['onset_utc']} breakup={row['breakup_utc']} "
          f"({row['n_changepoints']} cps)")
    return row


def write_results(rows: list[dict], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "changepoint_events.csv"
    cols = ["station", "winter", "encoder", "method", "signal",
            "onset_utc", "breakup_utc", "n_changepoints", "notes"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[cp] wrote {len(rows)} rows -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "dinov2"])
    ap.add_argument("--station")
    ap.add_argument("--winter")
    ap.add_argument("--all", action="store_true",
                    help="the two cedarburg train winters (PAPER_TRAIN_WINTERS)")
    ap.add_argument("--method", default="bocpd", choices=["bocpd", "beast"])
    ap.add_argument("--signal", default="pc1", choices=["pc1", "diffnorm"])
    ap.add_argument("--stem", default=None,
                    help="override the cache file stem (for smoke embeddings)")
    ap.add_argument("--beast-thresh", type=float, default=BEAST_CP_PROB_THRESH,
                    help="BEAST trend cpOccPr threshold (cpOccPr is diffuse; ~0.1)")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    targets = PAPER_TRAIN_WINTERS if args.all else [(args.station, args.winter)]
    if any(s is None or w is None for s, w in targets):
        raise SystemExit("Provide --station and --winter, or --all.")

    rows = [run_one(args.encoder, s, w, args.method, args.signal, stem=args.stem,
                    beast_thresh=args.beast_thresh)
            for s, w in targets]
    write_results(rows, Path(args.results))


def _self_test() -> None:
    """Synthetic check: a 1-D series with a known level shift -> BOCPD finds it."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
    x = (x - x.mean()) / (x.std() + 1e-9)
    res = bocpd_gaussian(x, hazard_lambda=120.0)
    cps = changepoints_from_runlength(res["map_run_length"], min_drop=12, min_sep=12)
    assert cps, "self-test: BOCPD found no change point in a series with a clear shift"
    nearest = min(cps, key=lambda i: abs(i - 200))
    assert abs(nearest - 200) <= 25, f"self-test: change point {nearest} far from 200"

    # diffnorm reduction on a synthetic embedding jump should also be non-degenerate.
    X = np.vstack([rng.normal(0, 0.1, (120, 8)), rng.normal(3, 0.1, (120, 8))])
    sig = reduce_signal(X.astype("float32"), "diffnorm")
    assert sig.shape[0] == X.shape[0], "diffnorm signal must stay aligned to clips"
    assert np.argmax(sig) == 120, "diffnorm should spike at the embedding jump"
    print("[cp] self-test OK: BOCPD nearest cp =", nearest, "(true=200); "
          "diffnorm spike at", int(np.argmax(sig)))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _self_test()
    else:
        main()
