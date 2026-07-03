"""GATE A (H3): do frozen FM embeddings separate ice from open water, and does
that separation survive changes in lighting?

This is the go/no-go probe between Phase 2 (embeddings) and Phase 3 (temporal
head). It fits a *linear* readout on top of the frozen embeddings and asks three
questions per station-winter:

  1. Pooled separability   AUC of a linear probe predicting the per-clip ice_flag,
                           scored out-of-fold with TIME-BLOCKED cross-validation
                           (random k-fold would leak: clips overlap by design, so
                           neighbours share frames).
  2. Lighting invariance   the same probe refit and scored WITHIN day-only and
                           WITHIN night-only clips. High day AUC is strong H3
                           evidence (ice and water are both well lit); night is
                           the hard case. Glare clips get the same treatment when
                           there are enough of them.
  3. Lighting confound     a baseline probe that sees ONLY lighting features
                           (night_frac, brightness_mean). If lighting alone
                           already predicts ice (long dark winter nights), the
                           embedding's advantage over it is the honest signal.

A linear probe is deliberate: we are testing the representation, not training a
classifier. If a linear readout already separates ice from water across lighting,
the temporal head has something real to model. If it does not, the fix is in
preprocessing (river crop, water mask, clip width) BEFORE modeling -- that is the
"Fail" branch of GATE A in IMPLEMENTATION_PLAN.md.

Everything here is CPU-only and reads the cached .npy embeddings + the labeled
clip manifest. No GPU, no network.

Usage:
  python -m fm_ice.evaluation.probe_separability --all                 # all paper winters, vjepa2
  python -m fm_ice.evaluation.probe_separability --station cedarburg --winter 2022-2023
  python -m fm_ice.evaluation.probe_separability --encoder dinov2 --all        # H2-side probe
  # validate the code path on the smoke embeddings before the full run lands:
  python -m fm_ice.evaluation.probe_separability --station cedarburg --winter 2022-2023 \
         --stem 2022-2023_smoke32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml

# Paper station-winters (same convention as evaluation/reference_events.py).
PAPER_WINTERS = [("cedarburg", "2022-2023"), ("cedarburg", "2023-2024"),
                 ("bismarck", "2024-2025")]

# A clip is "night" when most of its frames are after dark; "glare" when any frame
# is glared. Thresholds are deliberately simple and reported alongside the numbers.
NIGHT_FRAC_THRESH = 0.5
GLARE_FRAC_THRESH = 0.0      # strictly > 0  -> at least one glared frame

# Minimum support to even attempt an AUC (need both classes, enough points for CV).
MIN_PER_CLASS = 8
MIN_TOTAL = 40
N_FOLDS = 5

# GATE A verdict thresholds (reported, not enforced -- the human makes the call).
POOLED_AUC_PASS = 0.80       # strong pooled separability
DAY_AUC_PASS = 0.70          # separates ice/water in good light
NIGHT_AUC_REVIEW = 0.60      # below this at night = lighting is doing the work


def _paths():
    cfg = load_yaml("pipeline.yaml")
    return cfg["paths"], cfg


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _balanced_acc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, (score >= 0.5).astype(int)))


def _time_blocked_oof(X: np.ndarray, y: np.ndarray, order: np.ndarray,
                      k: int = N_FOLDS) -> np.ndarray:
    """Out-of-fold probe scores using k contiguous time blocks.

    `order` is the chronological rank of each row (argsort of t_start). Each fold
    is a contiguous time block held out while the probe trains on the rest, so a
    test clip is never predicted by a model that saw its temporal neighbour. The
    scaler is fit on the training fold only. Returns scores aligned to X's rows;
    rows in a fold whose train split is single-class get NaN.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    chrono = np.argsort(order, kind="stable")          # row indices in time order
    folds = np.array_split(chrono, k)
    for f in folds:
        test_idx = f
        train_idx = np.setdiff1d(chrono, test_idx, assume_unique=False)
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
            continue
        sc = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(X[train_idx]), y[train_idx])
        oof[test_idx] = clf.predict_proba(sc.transform(X[test_idx]))[:, 1]
    return oof


def _subgroup(X, y, order, mask, k=N_FOLDS):
    """Refit + score the probe WITHIN a subgroup (e.g. day-only). Returns a dict
    with n, n_pos, auc, balanced_acc -- or reason=None-filled if too small."""
    idx = np.where(mask)[0]
    pos = int(y[idx].sum())
    neg = int(len(idx) - pos)
    out = {"n": int(len(idx)), "n_pos": pos, "n_neg": neg,
           "auc": float("nan"), "bal_acc": float("nan")}
    if len(idx) < MIN_TOTAL or pos < MIN_PER_CLASS or neg < MIN_PER_CLASS:
        return out
    oof = _time_blocked_oof(X[idx], y[idx], order[idx], k=k)
    ok = ~np.isnan(oof)
    if ok.sum() >= MIN_TOTAL and len(np.unique(y[idx][ok])) == 2:
        out["auc"] = _auc(y[idx][ok], oof[ok])
        out["bal_acc"] = _balanced_acc(y[idx][ok], oof[ok])
    return out


def load_probe_matrix(encoder: str, station: str, winter: str, stem: str | None = None):
    """Load cached embeddings + the row-aligned index, join the labeled manifest.

    Returns (X, df) where X is (N, D) float32 and df is the manifest rows in the
    SAME order as X, carrying ice_flag and the QC lighting fractions. Raises with a
    clear message if the embedding cache is missing or rows are unlabeled.
    """
    paths, _ = _paths()
    stem = stem or winter
    cache = Path(paths["cache"]) / encoder / station
    npy = cache / f"{stem}.npy"
    idx_csv = cache / f"{stem}_index.csv"
    if not npy.exists():
        raise SystemExit(f"missing embeddings {npy} -- run extract_embeddings first")

    X = np.load(npy).astype("float32")
    index = pd.read_csv(idx_csv)
    manifest = pd.read_parquet(Path(paths["interim"]) / station / f"clips_{winter}.parquet")

    # Align manifest to the embedding row order via clip_id (the index is the
    # ground truth for which clip is which row).
    man = manifest.set_index("clip_id")
    if not index["clip_id"].isin(man.index).all():
        raise SystemExit("index clip_ids not all present in manifest -- stale cache?")
    df = man.loc[index["clip_id"].to_numpy()].reset_index()
    if len(df) != len(X):
        raise SystemExit(f"row mismatch: {len(X)} embeddings vs {len(df)} manifest rows")

    # Keep only clips with a definite ice label (drop NA Int64).
    keep = df["ice_flag"].notna().to_numpy()
    if not keep.all():
        X, df = X[keep], df.loc[keep].reset_index(drop=True)
    return X, df


def probe_one(encoder: str, station: str, winter: str, stem: str | None = None) -> dict:
    """Run the full GATE-A probe for one (station, winter). Returns a result row."""
    X, df = load_probe_matrix(encoder, station, winter, stem=stem)
    y = df["ice_flag"].astype(int).to_numpy()
    order = pd.to_datetime(df["t_start_utc"], utc=True).astype("int64").to_numpy()

    row = {"encoder": encoder, "station": station, "winter": winter,
           "n_clips": int(len(y)), "n_ice": int(y.sum()),
           "ice_rate": round(float(y.mean()), 3) if len(y) else float("nan")}

    if y.sum() < MIN_PER_CLASS or (len(y) - y.sum()) < MIN_PER_CLASS or len(y) < MIN_TOTAL:
        row["note"] = "insufficient labeled support for a probe"
        for key in ("pooled_auc", "pooled_bal_acc", "day_auc", "night_auc",
                    "glare_auc", "lighting_only_auc"):
            row[key] = float("nan")
        return row

    # 1. pooled, time-blocked
    oof = _time_blocked_oof(X, y, order)
    ok = ~np.isnan(oof)
    row["pooled_auc"] = round(_auc(y[ok], oof[ok]), 3)
    row["pooled_bal_acc"] = round(_balanced_acc(y[ok], oof[ok]), 3)

    # 2. lighting invariance
    night_frac = df["night_frac"].fillna(0).to_numpy()
    glare_frac = df["glare_frac"].fillna(0).to_numpy()
    day = _subgroup(X, y, order, night_frac <= NIGHT_FRAC_THRESH)
    night = _subgroup(X, y, order, night_frac > NIGHT_FRAC_THRESH)
    glare = _subgroup(X, y, order, glare_frac > GLARE_FRAC_THRESH)
    row["day_auc"] = round(day["auc"], 3)
    row["day_n"] = day["n"]
    row["night_auc"] = round(night["auc"], 3)
    row["night_n"] = night["n"]
    row["glare_auc"] = round(glare["auc"], 3)
    row["glare_n"] = glare["n"]

    # 3. lighting-only confound baseline (same CV, 2 lighting features only)
    light = np.column_stack([night_frac, df["brightness_mean"].fillna(
        df["brightness_mean"].median()).to_numpy()]).astype("float32")
    loof = _time_blocked_oof(light, y, order)
    lok = ~np.isnan(loof)
    row["lighting_only_auc"] = round(_auc(y[lok], loof[lok]), 3)
    row["embed_minus_lighting"] = round(row["pooled_auc"] - row["lighting_only_auc"], 3)

    # verdict (reported, human decides the gate)
    p, d, n = row["pooled_auc"], row["day_auc"], row["night_auc"]
    if p >= POOLED_AUC_PASS and (np.isnan(d) or d >= DAY_AUC_PASS):
        row["verdict"] = "PASS"
    elif p >= 0.70:
        row["verdict"] = "REVIEW"
    else:
        row["verdict"] = "FAIL"
    if not np.isnan(n) and n < NIGHT_AUC_REVIEW and row["verdict"] == "PASS":
        row["verdict"] = "PASS(day-driven)"
    return row


def run(encoder: str, targets, stem: str | None = None, results_dir: str = "results") -> list[dict]:
    rows = []
    for station, winter in targets:
        try:
            rows.append(probe_one(encoder, station, winter, stem=stem))
        except SystemExit as e:
            rows.append({"encoder": encoder, "station": station, "winter": winter,
                         "note": str(e)})
    _report(rows)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"gate_a_{encoder}.csv", index=False)
    (out / f"gate_a_{encoder}.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[gate-a] wrote {out / ('gate_a_' + encoder + '.csv')}")
    return rows


def _report(rows: list[dict]) -> None:
    print("\n================ GATE A (H3): FM ice/water separability ================")
    hdr = f"{'station-winter':22s} {'n(ice)':>11s} {'pooled':>7s} {'day':>6s} {'night':>6s} {'light-only':>11s} {'verdict':>16s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        sw = f"{r['station']}-{r['winter']}"
        if "pooled_auc" not in r or (isinstance(r.get("pooled_auc"), float) and np.isnan(r.get("pooled_auc", float('nan')))):
            print(f"{sw:22s}  {r.get('note','(no result)')}")
            continue
        nice = f"{r['n_clips']}({r['n_ice']})"
        print(f"{sw:22s} {nice:>11s} {r['pooled_auc']:>7.3f} {r['day_auc']:>6.3f} "
              f"{r['night_auc']:>6.3f} {r['lighting_only_auc']:>11.3f} {r.get('verdict',''):>16s}")
    print("\nAUC = out-of-fold, time-blocked 5-fold (clips overlap, so no random CV).")
    print("day/night refit within each lighting regime; light-only = night_frac+brightness baseline.")
    print(f"Heuristic verdict: PASS if pooled>={POOLED_AUC_PASS} & day>={DAY_AUC_PASS}; "
          f"REVIEW if pooled>=0.70; else FAIL. Human makes the gate call.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "dinov2"])
    ap.add_argument("--station")
    ap.add_argument("--winter")
    ap.add_argument("--all", action="store_true", help="all paper station-winters")
    ap.add_argument("--stem", default=None,
                    help="override the cache file stem (e.g. 2022-2023_smoke32 to dry-run)")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    targets = PAPER_WINTERS if args.all else [(args.station, args.winter)]
    if any(s is None or w is None for s, w in targets):
        raise SystemExit("Provide --station and --winter, or --all.")
    run(args.encoder, targets, stem=args.stem, results_dir=args.results)


if __name__ == "__main__":
    main()
