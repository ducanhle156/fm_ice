"""C2 smoke test: ice-jam candidate windows from the predictive entropy of the
trained temporal head (docs/FM_ice_plan_v2.md Sec. 6 + addendum item 4).

Physics of the signal: the head outputs p(ice) per clip. On open water or solid
sheet ice it is confident (H ~ 0); when the river is physically ambiguous --
piled, jammed, partial ice -- probability sits mid-range and the Bernoulli
entropy H(t) = -(p log p + (1-p) log(1-p)) spikes. Jams persist for days;
breakup ice runs and lighting artifacts last hours. Hence the 24 h persistence
rule below.

Probability source: the per-clip dumps results/phase3_pred_<enc>_<station>_
<winter>_<head>.csv (no retraining; no checkpoints exist). The cedarburg dumps
are OUT-OF-FOLD (leave-one-winter-out test predictions), which is the right
distribution for calibrating tau_H: it matches deployment-time entropy on an
unseen winter, whereas in-fold probabilities would be overconfident and push
tau_H down. Residual circularity to state in the paper: on the calibration
winters themselves ~1% of clips exceed tau_H by construction, so cedarburg's
false-alarm rows are optimistic-by-design; the transfer station (bismarck) and
future stations are the honest FAR rows.

Detector (all parameters frozen BEFORE any event station is touched):
  tau_H       99th percentile of H pooled over the cedarburg TRAIN winters.
  candidate   >= MIN_CLIPS consecutive above-threshold clips (24 h at the 4 h
              stride), where "consecutive" chains above-threshold clips whose
              gap to the previous above-threshold clip is < MAX_GAP_H. The one
              rule implements both "6 consecutive clips" and "merge gaps < 8 h",
              and works on real timestamps (QC leaves gaps up to 180 h).

Outputs (results/entropy_jam/):
  <station>_<winter>_entropy.npy   float64 (T,) H, row-aligned to the pred CSV
  detections.json                  frozen params + per-station-winter windows
  false_alarm_table.csv            FAR per station-winter (no documented jams in
                                   the cached winters: every detection counts as
                                   a false alarm; the deliverable is the FAR)

Usage:
  python -m fm_ice.evaluation.entropy_jam                        # vjepa2 tcn
  python -m fm_ice.evaluation.entropy_jam --encoder dinov2       # robustness rerun
  python -m fm_ice.evaluation.entropy_jam --tau-h 0.6            # sensitivity check

Self-test (no data): python -m fm_ice.evaluation.entropy_jam --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.evaluation.reextract_events import PRED_RE

EPS = 1e-8
MIN_CLIPS = 6          # >= 6 above-threshold clips = 24 h at the 4 h stride
MAX_GAP_H = 8.0        # a longer silence between above-threshold clips breaks a window
TAU_Q = 99.0           # percentile of H on the calibration winters
CALIB_STATION = "cedarburg"


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Bernoulli predictive entropy in nats (max ln 2 ~ 0.693)."""
    p = np.asarray(p, dtype=float)
    return -(p * np.log(p + EPS) + (1.0 - p) * np.log(1.0 - p + EPS))


def entropy_windows(times_utc, H: np.ndarray, tau_h: float,
                    min_clips: int = MIN_CLIPS, max_gap_h: float = MAX_GAP_H,
                    prob: np.ndarray | None = None) -> list[dict]:
    """Candidate jam windows from an entropy series on REAL timestamps."""
    t = pd.to_datetime(pd.Series(list(times_utc)), utc=True).reset_index(drop=True)
    H = np.asarray(H, dtype=float)
    above = np.flatnonzero(H > tau_h)
    windows: list[list[int]] = []
    for idx in above:
        if windows and (t[idx] - t[windows[-1][-1]]) < pd.Timedelta(hours=max_gap_h):
            windows[-1].append(int(idx))
        else:
            windows.append([int(idx)])
    out = []
    for w in windows:
        if len(w) < min_clips:
            continue
        rec = {"start_utc": str(t[w[0]]), "end_utc": str(t[w[-1]]),
               "n_clips": len(w),
               "duration_h": round((t[w[-1]] - t[w[0]]).total_seconds() / 3600, 1),
               "max_H": round(float(H[w].max()), 4),
               "mean_H": round(float(H[w].mean()), 4)}
        if prob is not None:
            rec["mean_prob"] = round(float(np.asarray(prob, dtype=float)[w].mean()), 4)
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
def discover_preds(results_dir: Path, encoder: str, head: str) -> list[dict]:
    out = []
    for f in sorted(results_dir.glob(f"phase3_pred_{encoder}_*_{head}.csv")):
        m = PRED_RE.match(f.name)
        if m:
            out.append({**m.groupdict(), "path": f})
    return out


def calibrate_tau_h(preds: list[dict], calib_station: str, q: float) -> tuple[float, list[str]]:
    """tau_H = q-th percentile of H pooled over the calibration station's winters
    (out-of-fold LOO probabilities). Frozen before any event station is scored."""
    calib = [p for p in preds if p["station"] == calib_station]
    if not calib:
        raise SystemExit(f"no {calib_station} pred dumps to calibrate tau_H on")
    Hs = [binary_entropy(pd.read_csv(p["path"])["prob_temporal_head"].to_numpy())
          for p in calib]
    return float(np.percentile(np.concatenate(Hs), q)), sorted(p["winter"] for p in calib)


def run(results_dir: Path, encoder: str, head: str, tau_h: float | None) -> None:
    preds = discover_preds(results_dir, encoder, head)
    if not preds:
        raise SystemExit(f"no phase3_pred_{encoder}_*_{head}.csv dumps in {results_dir}")
    out_dir = results_dir / "entropy_jam"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_note = "manual override (--tau-h)"
    calib_winters: list[str] = []
    if tau_h is None:
        tau_h, calib_winters = calibrate_tau_h(preds, CALIB_STATION, TAU_Q)
        calib_note = (f"p{TAU_Q:.0f} of H pooled over {CALIB_STATION} "
                      f"{'+'.join(calib_winters)} (out-of-fold LOO probs), frozen")
    print(f"[entropy_jam] tau_H = {tau_h:.4f}  ({calib_note})")

    detections: dict[str, list] = {}
    far_rows = []
    for p in preds:
        df = pd.read_csv(p["path"])
        times = pd.to_datetime(df["t_start_utc"], utc=True)
        prob = df["prob_temporal_head"].to_numpy(dtype=float)
        H = binary_entropy(prob)
        key = f"{p['station']}_{p['winter']}"
        np.save(out_dir / f"{key}_entropy.npy", H)

        wins = entropy_windows(times, H, tau_h, prob=prob)
        detections[key] = wins
        span_days = (times.iloc[-1] - times.iloc[0]).total_seconds() / 86400
        flagged = int((H > tau_h).sum())
        far_rows.append({
            "station": p["station"], "winter": p["winter"],
            "n_clips": len(df), "span_days": round(span_days, 1),
            "n_detections": len(wins),
            "flagged_clips": flagged,
            "flagged_hours": round(sum(w["duration_h"] for w in wins), 1),
            "detections_per_100d": round(100 * len(wins) / span_days, 2),
            "note": "no documented jam this station-winter; every detection "
                    "counts as a false alarm",
        })
        print(f"  {key:24s} clips={len(df):5d}  H>tau: {flagged:4d}  "
              f"detections: {len(wins)}")
        for w in wins:
            print(f"    {w['start_utc']} .. {w['end_utc']}  "
                  f"({w['n_clips']} clips, {w['duration_h']} h, "
                  f"mean_p={w['mean_prob']})")

    meta = {"encoder": encoder, "head": head, "tau_H": round(tau_h, 6),
            "tau_calibration": calib_note, "tau_percentile": TAU_Q,
            "calib_station": CALIB_STATION, "calib_winters": calib_winters,
            "min_clips": MIN_CLIPS, "max_gap_h": MAX_GAP_H, "entropy_eps": EPS,
            "source_csvs": [p["path"].name for p in preds]}
    (out_dir / "detections.json").write_text(
        json.dumps({"meta": meta, "detections": detections}, indent=2))
    far = pd.DataFrame(far_rows)
    far.to_csv(out_dir / "false_alarm_table.csv", index=False)
    print(f"\n===== false-alarm table (anchor winters, no jams expected) =====")
    print(far.drop(columns=["note"]).to_string(index=False))
    print(f"\n[entropy_jam] wrote {out_dir}/detections.json, false_alarm_table.csv, "
          f"{len(preds)} entropy .npy files")


def _selftest() -> None:
    # binary entropy: symmetric, 0 at the ends, ln 2 at 0.5.
    H = binary_entropy(np.array([0.0, 0.5, 1.0]))
    assert H[1] > H[0] and H[1] > H[2]
    assert abs(H[1] - np.log(2)) < 1e-6 and H[0] < 1e-6

    # windowing on real timestamps: 4 h stride, tau 0.5.
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    times = [t0 + pd.Timedelta(hours=4 * i) for i in range(40)]
    H = np.zeros(40)
    H[5:12] = 0.6          # 7 consecutive clips -> detection
    H[20:23] = 0.6         # 3 clips -> too short
    H[30] = 0.6            # isolated clip -> no
    wins = entropy_windows(times, H, 0.5)
    assert len(wins) == 1 and wins[0]["n_clips"] == 7, wins
    assert wins[0]["start_utc"].startswith("2025-01-01 20:00")

    # a 4 h gap (one below-threshold clip) does NOT break a window (< 8 h)...
    H2 = np.zeros(40); H2[5:9] = 0.6; H2[10:13] = 0.6     # gap at idx 9 = 8 h apart
    wins2 = entropy_windows(times, H2, 0.5)
    assert len(wins2) == 0, wins2       # 8 h gap is NOT < 8 h: two short windows
    H4 = np.zeros(40); H4[5:9] = 0.6; H4[9] = 0.55; H4[10:13] = 0.6
    wins4 = entropy_windows(times, H4, 0.5)
    assert len(wins4) == 1 and wins4[0]["n_clips"] == 8, wins4

    # ...but a QC hole in the TIMESTAMPS >= 8 h splits the chain.
    times_gap = list(times)
    for i in range(10, 40):
        times_gap[i] = times[i] + pd.Timedelta(hours=48)
    H5 = np.zeros(40); H5[6:14] = 0.6
    wins5 = entropy_windows(times_gap, H5, 0.5)
    assert len(wins5) == 0, wins5       # 4+4 clips split by the 52 h hole
    print("entropy_jam self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "dinov2"])
    ap.add_argument("--head", default="tcn", choices=["tcn", "transformer"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--tau-h", type=float, default=None,
                    help="manual tau_H override for sensitivity checks "
                         "(default: calibrate on cedarburg train winters, frozen)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    run(Path(args.results), args.encoder, args.head, args.tau_h)


if __name__ == "__main__":
    main()
