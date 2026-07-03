"""Phase 5 -- transfer test: score the temporal head on the held-out STATION.

Phase 3 trained and scored the head LEAVE-ONE-WINTER-OUT on the train station
(cedarburg). Phase 5 asks the harder question: how well does a head trained ONLY
on cedarburg generalize to an entirely unseen station -- bismarck 2024-2025 --
with NO retraining and NO feature scaling fit on bismarck. The contribution is a
measured *transfer gap*: bismarck timing error minus the cedarburg held-out
(leave-one-winter-out) timing error.

Protocol (no bismarck leakage):
  train_on_cedarburg   load BOTH cedarburg winters (2022-2023 + 2023-2024) as two
                       training sequences, fit the FeatureScaler on those winters
                       ONLY, train the TemporalHead, and fit the per-frame logistic
                       probe (the transfer anchor) on the same cedarburg features.
  evaluate_transfer    transform bismarck 2024-2025 with the cedarburg-fit scaler,
                       predict with the frozen head + probe, read onset/breakup
                       (optionally with the physical freezing guard), and score
                       timing error against the bismarck references. The scaler and
                       the head never see a single bismarck frame.

The cedarburg held-out anchor is read from results/phase3_summary.csv when present.
If that file is absent we fall back to the Phase-3 timing CSV
(results/phase3_timing_<encoder>_<head>.csv), averaging the temporal-head
onset/breakup error vs the USGS ice flag across the leave-one-winter-out folds.
If neither exists the gap is reported as unavailable.

Outputs (results/):
  phase5_transfer_<encoder>_<head>[_guard].csv   per (model, reference) timing + the transfer gap.
  phase3_pred_<encoder>_<station>_<winter>_<head>[_guard].csv
                                                 per-step bismarck prob/state for head + probe,
                                                 written under the CANONICAL phase3_pred schema so the
                                                 Phase-6 evaluators (fm_ice.evaluation.evaluate and
                                                 error_analysis) score the held-out station automatically.

Usage:
  python -m fm_ice.models.transfer --encoder vjepa2 --temp-guard
  python -m fm_ice.models.transfer --encoder vjepa2 --head transformer
  python -m fm_ice.models.transfer --encoder dinov2 --epochs 80
  python -m fm_ice.models.transfer --selftest        # anchor parser self-test, no data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml
from fm_ice.models.events import min_run_steps_from_hours, read_events
from fm_ice.models.train import (
    FeatureScaler,
    evaluate_prediction,
    fit_head,
    fit_perframe,
    load_sequence,
    predict_head,
)


# --------------------------------------------------------------------------- #
# Train on the train station (cedarburg) only -- no bismarck anywhere here.
# --------------------------------------------------------------------------- #
def _train_station(stations_cfg: dict) -> str:
    """The station whose role is 'train' (cedarburg). Config over hardcoding."""
    for name, s in stations_cfg["stations"].items():
        if s.get("role") == "train":
            return name
    raise SystemExit("no station with role 'train' in stations.yaml")


def _train_winters(encoder: str, station: str) -> list[str]:
    """All cedarburg winters with a cached embedding for this encoder, sorted."""
    cfg = load_yaml("pipeline.yaml")
    cache = Path(cfg["paths"]["cache"]) / encoder / station
    winters = sorted(
        p.stem for p in cache.glob("*.npy") if "_smoke" not in p.stem and "_index" not in p.stem
    )
    if not winters:
        raise SystemExit(f"no cached embeddings under {cache} -- run extract_embeddings first")
    return winters


def train_on_cedarburg(encoder: str, head: str, epochs: int, cfg: dict):
    """Train the temporal head + per-frame probe on ALL cedarburg winters together.

    Returns (model, scaler, perframe_clf). The scaler is fit on the cedarburg
    winters only -- this is the no-leakage guarantee for the transfer test.
    """
    th = cfg["temporal_head"]
    seed = cfg["seed"]
    np.random.seed(seed)

    stations_cfg = load_yaml("stations.yaml")
    station = _train_station(stations_cfg)
    winters = _train_winters(encoder, station)

    print(f"\n=== train station={station} | winters={winters} | encoder={encoder} | head={head} ===")
    tr_emb, tr_air, tr_y = [], [], []
    for w in winters:
        e, a, y, _ = load_sequence(encoder, station, w)
        tr_emb.append(e)
        tr_air.append(a)
        tr_y.append(y)
        print(f"  loaded {station} {w}: T={len(y)}  ice_steps={int(y.sum())}")

    scaler = FeatureScaler(th["use_air_temp"]).fit(tr_emb, tr_air)
    tr_feats = [scaler.transform(e, a) for e, a in zip(tr_emb, tr_air)]

    model = fit_head(tr_feats, tr_y, scaler.out_dim, th, epochs, head, seed)
    clf = fit_perframe(tr_feats, tr_y)
    return model, scaler, clf


# --------------------------------------------------------------------------- #
# The cedarburg held-out anchor for the gap.
# --------------------------------------------------------------------------- #
def _cedarburg_loo_anchor(encoder: str, head: str, temp_guard: bool, results_dir: Path) -> dict:
    """Cedarburg leave-one-winter-out temporal-head timing error (the gap anchor).

    Prefers results/phase3_summary.csv (the consolidated H1 table from
    fm_ice.evaluation.phase3_report): one row per config, e.g.

        config                                2022-2023 onset  2022-2023 breakup  ...  mean_err_h
        vjepa2 / tcn / temporal-head +guard   446.0            2.0                ...  122.0

    We select the temporal-head row matching THIS encoder/head/guard exactly,
    average the per-winter 'onset'/'breakup' columns across the leave-one-out
    folds, and read 'mean_err_h'. Falls back to the per-config Phase-3 timing CSV
    (phase3_timing_<encoder>_<head>[_guard].csv), averaging temporal_head error
    vs usgs_ice_flag across folds. Returns {'onset','breakup','mean','source'}
    with None values when unavailable.
    """
    none = {"onset": None, "breakup": None, "mean": None, "source": "unavailable"}

    summary = results_dir / "phase3_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        # The config string is written by phase3_report as
        # "<encoder> / <head> / temporal-head[ +guard]"; match it EXACTLY so a
        # no-guard request does not accidentally pull the "+guard" row (or the
        # per-frame-probe rows that share the encoder/head substring).
        want = f"{encoder} / {head} / temporal-head" + (" +guard" if temp_guard else "")
        if "config" in df.columns:
            sub = df[df["config"].astype(str) == want]
            if not sub.empty:
                r = sub.iloc[0]
                on = _mean(*[r[c] for c in df.columns if c.endswith(" onset") and pd.notna(r[c])])
                br = _mean(*[r[c] for c in df.columns if c.endswith(" breakup") and pd.notna(r[c])])
                mn = (float(r["mean_err_h"]) if "mean_err_h" in df.columns and pd.notna(r["mean_err_h"])
                      else _mean(on, br))
                return {"onset": on, "breakup": br, "mean": mn, "source": "phase3_summary.csv"}

    # Fallback: the per-config Phase-3 timing CSV written by models.train. The
    # guard variant is a distinct file, so selecting the right filename applies it.
    guard_suffix = "_guard" if temp_guard else ""
    timing = results_dir / f"phase3_timing_{encoder}_{head}{guard_suffix}.csv"
    if timing.exists():
        df = pd.read_csv(timing)
        sub = df[(df["model"] == "temporal_head") & (df["reference"] == "usgs_ice_flag")]
        if "head" in sub.columns:
            sub = sub[sub["head"] == head]
        if not sub.empty:
            on = float(sub["onset_err_h"].dropna().mean()) if sub["onset_err_h"].notna().any() else None
            br = float(sub["breakup_err_h"].dropna().mean()) if sub["breakup_err_h"].notna().any() else None
            return {"onset": on, "breakup": br, "mean": _mean(on, br), "source": timing.name}

    return none


def _mean(*vals):
    nums = [v for v in vals if v is not None]
    return float(np.mean(nums)) if nums else None


def _round(v):
    return round(v, 1) if v is not None else None


# --------------------------------------------------------------------------- #
# Evaluate transfer on bismarck.
# --------------------------------------------------------------------------- #
def evaluate_transfer(encoder: str, head: str, epochs: int, temp_guard: bool,
                      cfg: dict, results_dir: Path) -> pd.DataFrame:
    """Predict bismarck 2024-2025 with the cedarburg-trained head + probe and
    report timing error and the transfer gap vs the cedarburg held-out anchor."""
    ev_cfg = cfg["events"]
    threshold = ev_cfg["ice_state_threshold"]
    min_run = min_run_steps_from_hours(ev_cfg["min_event_separation_hours"], cfg["clip"]["stride_hours"])
    guard_c = ev_cfg.get("onset_freeze_guard_c", 0.0) if temp_guard else None
    guard_suffix = "_guard" if temp_guard else ""   # disambiguates the matrix runs

    stations_cfg = load_yaml("stations.yaml")
    splits = stations_cfg["splits"]
    station = splits["transfer_station"]      # bismarck
    winter = splits["test_winter"]            # 2024-2025

    # --- train on cedarburg only (scaler + head + probe) ---
    model, scaler, clf = train_on_cedarburg(encoder, head, epochs, cfg)

    # --- transfer: transform bismarck with the cedarburg-fit scaler (no leakage) ---
    print(f"\n=== transfer test station={station} {winter} | encoder={encoder} | "
          f"head={head} | temp_guard={temp_guard} ===")
    te_emb, te_air, te_y, te_times = load_sequence(encoder, station, winter)
    te_feat = scaler.transform(te_emb, te_air)
    print(f"  bismarck {winter}: T={len(te_y)}  ice_steps={int(te_y.sum())}")

    prob_head = predict_head(model, te_feat)
    prob_pf = clf.predict_proba(te_feat)[:, 1]

    anchor = _cedarburg_loo_anchor(encoder, head, temp_guard, results_dir)

    # --- per-step prediction dump ---
    pred_dump = {
        "t_start_utc": te_times.astype(str),
        "ice_flag": te_y,
        "air_tmpc": te_air,
    }

    rows = []
    for name, prob in (("temporal_head", prob_head), ("perframe_probe", prob_pf)):
        ev = read_events(te_times, prob, threshold=threshold, min_run_steps=min_run,
                         air=te_air, max_run_mean_air=guard_c)
        print(f"  [{name:14s}] onset={ev['onset']}  breakup={ev['breakup']}  ice_steps={ev['n_ice_steps']}")
        pred_dump[f"prob_{name}"] = prob
        pred_dump[f"state_{name}"] = ev["state"]

        scored = evaluate_prediction(ev, station, winter, results_dir)
        for r in scored:
            b_on = r["onset_err_h"]
            b_br = r["breakup_err_h"]
            # The gap is defined on the temporal head (the contribution) against
            # the USGS-flag reference only -- the cedarburg LOO anchor is computed
            # vs the same reference, so any other reference would be apples-to-oranges.
            # The per-frame probe rows are reported as the transfer anchor for context.
            gap = None
            if (name == "temporal_head" and r["reference"] == "usgs_ice_flag"
                    and anchor["mean"] is not None):
                gap = _mean(
                    (b_on - anchor["onset"]) if (b_on is not None and anchor["onset"] is not None) else None,
                    (b_br - anchor["breakup"]) if (b_br is not None and anchor["breakup"] is not None) else None,
                )
            rows.append({
                "encoder": encoder,
                "model": name,
                "head": head,
                "temp_guard": temp_guard,
                "reference": r["reference"],
                "onset_err_h": _round(b_on),
                "breakup_err_h": _round(b_br),
                "bismarck_onset": str(ev["onset"]) if ev["onset"] is not None else None,
                "bismarck_breakup": str(ev["breakup"]) if ev["breakup"] is not None else None,
                "cedarburg_loo_mean": _round(anchor["mean"]),
                "transfer_gap_h": _round(gap),
            })

    df = pd.DataFrame(rows)
    out_csv = results_dir / f"phase5_transfer_{encoder}_{head}{guard_suffix}.csv"
    df.to_csv(out_csv, index=False)
    # Write the per-step dump under the CANONICAL phase3_pred schema so the Phase-6
    # evaluators pick up the held-out station with no code change on their side.
    pred_csv = results_dir / f"phase3_pred_{encoder}_{station}_{winter}_{head}{guard_suffix}.csv"
    pd.DataFrame(pred_dump).to_csv(pred_csv, index=False)

    _report(df, encoder, head, anchor)
    print(f"\n[phase5] wrote {out_csv}")
    print(f"[phase5] wrote {pred_csv}")
    return df


def _report(df: pd.DataFrame, encoder: str, head: str, anchor: dict) -> None:
    print("\n================ Phase 5 (transfer): bismarck timing error (hours) ================")
    hdr = f"{'model':15s} {'reference':16s} {'onset_err':>9s} {'breakup_err':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        oe = f"{r['onset_err_h']:.1f}" if pd.notna(r["onset_err_h"]) else "  -"
        be = f"{r['breakup_err_h']:.1f}" if pd.notna(r["breakup_err_h"]) else "  -"
        print(f"{r['model']:15s} {r['reference']:16s} {oe:>9s} {be:>11s}")

    print(f"\nCedarburg held-out (LOO) anchor [{anchor['source']}]: "
          f"onset={_fmt(anchor['onset'])}  breakup={_fmt(anchor['breakup'])}  mean={_fmt(anchor['mean'])}")

    # Headline gap: temporal head vs the USGS-flag reference (same reference the
    # anchor is computed against).
    usgs = df[(df["model"] == "temporal_head") & (df["reference"] == "usgs_ice_flag")]
    if anchor["mean"] is None or usgs.empty:
        print("\nTRANSFER GAP: unavailable (no cedarburg LOO anchor on disk -- run Phase 3 first).")
    else:
        row = usgs.iloc[0]
        b_on = float(row["onset_err_h"]) if pd.notna(row["onset_err_h"]) else None
        b_br = float(row["breakup_err_h"]) if pd.notna(row["breakup_err_h"]) else None
        print(f"Bismarck temporal_head (vs usgs_ice_flag): "
              f"onset={_fmt(b_on)}  breakup={_fmt(b_br)}")
        on_gap = _delta(b_on, anchor["onset"])
        br_gap = _delta(b_br, anchor["breakup"])
        print("\nTRANSFER GAP (bismarck - cedarburg LOO, hours; positive = worse on transfer; vs usgs_ice_flag):")
        print(f"  onset  : {_fmt(on_gap)}")
        print(f"  breakup: {_fmt(br_gap)}")
        print(f"  mean   : {_fmt(_mean(on_gap, br_gap))}")

    print("\nNo Bismarck leakage: the FeatureScaler and the temporal head were fit on")
    print("cedarburg winters ONLY; bismarck was transformed and predicted, never trained on.")


def _delta(a, b):
    if a is None or b is None:
        return None
    if (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return None
    return a - b


def _fmt(v):
    return f"{v:.1f}h" if v is not None else "n/a"


# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Cheap self-test: the cedarburg LOO anchor parser (no data, no network)."""
    import tempfile

    summary = pd.DataFrame([
        {"config": "vjepa2 / tcn / temporal-head +guard",
         "2022-2023 onset": 446.0, "2022-2023 breakup": 2.0,
         "2023-2024 onset": 18.0, "2023-2024 breakup": 22.0, "mean_err_h": 122.0},
        {"config": "vjepa2 / tcn / temporal-head",
         "2022-2023 onset": 1146.0, "2022-2023 breakup": 2.0,
         "2023-2024 onset": 18.0, "2023-2024 breakup": 22.0, "mean_err_h": 297.0},
        {"config": "vjepa2 / tcn / perframe-probe",
         "2022-2023 onset": 726.0, "2022-2023 breakup": 70.0,
         "2023-2024 onset": 1122.0, "2023-2024 breakup": 58.0, "mean_err_h": 494.0},
    ])
    with tempfile.TemporaryDirectory() as d:
        rd = Path(d)
        summary.to_csv(rd / "phase3_summary.csv", index=False)
        g = _cedarburg_loo_anchor("vjepa2", "tcn", True, rd)
        assert g["source"] == "phase3_summary.csv", g
        assert g["mean"] == 122.0, g
        assert g["onset"] == 232.0, g          # mean(446, 18)
        assert g["breakup"] == 12.0, g         # mean(2, 22)
        # no-guard must pick the no-guard temporal-head row, not the +guard or
        # the per-frame-probe rows that share the "vjepa2"/"tcn" substring.
        ng = _cedarburg_loo_anchor("vjepa2", "tcn", False, rd)
        assert ng["mean"] == 297.0, ng
        assert ng["onset"] == 582.0, ng        # mean(1146, 18)
        # missing config -> unavailable, not a crash.
        miss = _cedarburg_loo_anchor("dinov2", "transformer", True, rd)
        assert miss["source"] == "unavailable" and miss["mean"] is None, miss
    print("transfer anchor self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "dinov2"])
    ap.add_argument("--head", default=None, choices=["tcn", "transformer"],
                    help="override temporal_head.type from the config")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--temp-guard", action="store_true",
                    help="apply the physical freezing guard (events.onset_freeze_guard_c)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--selftest", action="store_true",
                    help="run the cheap anchor-parsing self-test and exit")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    cfg = load_yaml("pipeline.yaml")
    head = args.head or cfg["temporal_head"]["type"]
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_transfer(args.encoder, head, args.epochs, args.temp_guard, cfg, results_dir)


if __name__ == "__main__":
    main()
