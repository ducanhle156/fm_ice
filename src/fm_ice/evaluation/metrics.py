"""Event-timing metrics for onset/breakup detection.

Two event types per winter: onset and breakup. We score predicted event times
against one or more references (USGS ice flag, stage breakpoint).

Implemented:
  timing_error_hours      absolute hours between matched predicted/reference events
  event_f1                precision/recall/F1 with a +/- tolerance window
  event_f1_at_tolerances  event_f1 looped over several tolerance windows
  covering                van den Burg & Williams covering metric over a segmentation
  per_frame_agreement     per-step ice-state agreement (accuracy/balanced/F1)
  per_frame_auc           per-step ice probability ROC-AUC

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


def event_f1_at_tolerances(pred_times: list, ref_times: list,
                           tolerances: list[float]) -> dict[float, PRF]:
    """Run event_f1 at several tolerance windows.

    Returns {tolerance_hours: PRF}, one matching pass per window. Used to build the
    24/48/72 h event-F1 sweep the plan asks for. Times are tz-aware UTC.
    """
    return {float(tol): event_f1(pred_times, ref_times, float(tol))
            for tol in tolerances}


def per_frame_agreement(pred_state, true_flag) -> dict[str, float]:
    """Per-step ice-state agreement: accuracy, balanced accuracy, and F1.

    pred_state / true_flag are 0/1 arrays of the SAME length (one entry per clip
    step). Rows where either side is NA are dropped first. Balanced accuracy and
    F1 degrade gracefully to NaN when only one class is present (a winter the head
    -- or the truth -- calls all-open or all-ice), since they are undefined there;
    plain accuracy is always reported.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y = pd.Series(list(true_flag))
    p = pd.Series(list(pred_state))
    keep = y.notna() & p.notna()
    y = y[keep].astype(int).to_numpy()
    p = p[keep].astype(int).to_numpy()
    out = {"accuracy": float("nan"), "balanced_acc": float("nan"), "f1": float("nan"),
           "n": int(len(y))}
    if len(y) == 0:
        return out
    out["accuracy"] = float(accuracy_score(y, p))
    n_true_classes = len(np.unique(y))
    if n_true_classes < 2:
        # balanced_acc and (positive-class) F1 are undefined / degenerate.
        return out
    out["balanced_acc"] = float(balanced_accuracy_score(y, p))
    out["f1"] = float(f1_score(y, p, zero_division=0))
    return out


def per_frame_auc(true_flag, prob) -> float:
    """Per-step ROC-AUC of the ice probability against the true flag.

    Returns NaN when only one class is present in true_flag (AUC undefined) or
    when there are no valid rows. NA rows on either side are dropped.
    """
    from sklearn.metrics import roc_auc_score

    y = pd.Series(list(true_flag))
    s = pd.Series(list(prob))
    keep = y.notna() & s.notna()
    y = y[keep].astype(int).to_numpy()
    s = s[keep].astype(float).to_numpy()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


if __name__ == "__main__":
    # Tiny self-test (no data needed).
    o_pred = pd.Timestamp("2025-01-05 06:00", tz="UTC")
    o_ref = pd.Timestamp("2025-01-04 12:00", tz="UTC")
    print("timing_error:", timing_error_hours({"onset": o_pred}, {"onset": o_ref}))
    prf = event_f1([o_pred], [o_ref], tolerance_hours=48)
    print("event_f1@48h:", prf)
    print("covering:", round(covering([4, 9], [5, 9], 12), 3))

    # event_f1_at_tolerances: a ~36 h gap matches at 48/72 h but not at 24 h.
    o_far = pd.Timestamp("2025-01-06 00:00", tz="UTC")   # 36 h from o_ref
    swept = event_f1_at_tolerances([o_far], [o_ref], [24, 48, 72])
    assert swept[24.0].tp == 0 and swept[48.0].tp == 1 and swept[72.0].tp == 1
    print("event_f1_at_tolerances:", {t: r.f1 for t, r in swept.items()})

    # per-frame agreement on a tiny two-class sequence.
    true = [0, 0, 1, 1, 1, 0]
    pred = [0, 1, 1, 1, 0, 0]
    agr = per_frame_agreement(pred, true)
    assert agr["n"] == 6 and 0.0 <= agr["accuracy"] <= 1.0
    assert not np.isnan(agr["balanced_acc"]) and not np.isnan(agr["f1"])
    print("per_frame_agreement:", {k: round(v, 3) if isinstance(v, float) else v
                                    for k, v in agr.items()})

    # single-class truth -> balanced_acc / f1 are NaN but accuracy is defined.
    agr1 = per_frame_agreement([0, 0, 0], [0, 0, 0])
    assert agr1["accuracy"] == 1.0 and np.isnan(agr1["balanced_acc"])

    # per-frame AUC: a probability that ranks ice above open water -> AUC 1.0.
    prob = [0.1, 0.2, 0.8, 0.9, 0.7, 0.3]
    auc = per_frame_auc(true, prob)
    assert abs(auc - 1.0) < 1e-9, auc
    assert np.isnan(per_frame_auc([1, 1, 1], [0.2, 0.5, 0.9]))  # one class -> NaN
    print("per_frame_auc:", round(auc, 3))
    print("metrics self-test OK")
