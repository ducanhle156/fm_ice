"""Event-timing metrics for onset/breakup detection.

Two event types per winter: onset and breakup. We score predicted event times
against one or more references (USGS ice flag, stage breakpoint).

Implemented:
  timing_error_hours   absolute hours between matched predicted/reference events
  event_f1             precision/recall/F1 with a +/- tolerance window
  covering             van den Burg & Williams covering metric over a segmentation

All event times are timezone-aware UTC datetimes or pandas Timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _hours(a, b) -> float:
    return abs((pd.Timestamp(a) - pd.Timestamp(b)).total_seconds()) / 3600.0


def timing_error_hours(pred: dict[str, object], ref: dict[str, object]) -> dict[str, float]:
    """Absolute timing error in hours per event type present in both dicts.

    pred/ref are like {'onset': Timestamp, 'breakup': Timestamp}. Missing keys
    are skipped. Returns {'onset': h, 'breakup': h, 'mean': h}.
    """
    out = {}
    for k in ("onset", "breakup"):
        if pred.get(k) is not None and ref.get(k) is not None:
            out[k] = _hours(pred[k], ref[k])
    if out:
        out["mean"] = float(np.mean(list(out.values())))
    return out


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def event_f1(pred_times: list, ref_times: list, tolerance_hours: float) -> PRF:
    """Greedy one-to-one matching within a tolerance window.

    A predicted event matches a reference event if within +/- tolerance_hours.
    Each reference is matched at most once (closest predicted wins).
    """
    preds = sorted(pd.Timestamp(t) for t in pred_times)
    refs = sorted(pd.Timestamp(t) for t in ref_times)
    used = [False] * len(refs)
    tp = 0
    for p in preds:
        best_j, best_d = -1, None
        for j, r in enumerate(refs):
            if used[j]:
                continue
            d = _hours(p, r)
            if d <= tolerance_hours and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fp = len(preds) - tp
    fn = len(refs) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def covering(true_segends: list[int], pred_segends: list[int], n: int) -> float:
    """Covering metric (van den Burg & Williams 2020) for a 1-D segmentation of
    length n. Segment ends are indices that close a segment (inclusive)."""
    def to_segments(ends):
        segs, start = [], 0
        for e in sorted(set(ends + [n - 1])):
            segs.append(set(range(start, e + 1)))
            start = e + 1
        return [s for s in segs if s]

    G = to_segments(true_segends)
    P = to_segments(pred_segends)
    total = 0.0
    for A in G:
        best = max((len(A & B) / len(A | B) for B in P), default=0.0)
        total += len(A) * best
    return total / n


if __name__ == "__main__":
    # Tiny self-test (no data needed).
    o_pred = pd.Timestamp("2025-01-05 06:00", tz="UTC")
    o_ref = pd.Timestamp("2025-01-04 12:00", tz="UTC")
    print("timing_error:", timing_error_hours({"onset": o_pred}, {"onset": o_ref}))
    prf = event_f1([o_pred], [o_ref], tolerance_hours=48)
    print("event_f1@48h:", prf)
    print("covering:", round(covering([4, 9], [5, 9], 12), 3))
