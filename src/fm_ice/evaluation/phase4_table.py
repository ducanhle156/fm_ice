"""Phase-4 H2 head-to-head timing table + GATE B verdict.

This is the central H2 figure for a CV audience: ONE table comparing every method
on onset/breakup timing error against the solid USGS ice-flag reference, per
held-out cedarburg winter (leave-one-winter-out) plus a mean.

Rows (methods), TCN head family, guard on and off shown side by side:
  V-JEPA temporal (TCN)          temporal head on vjepa2, no freezing guard
  V-JEPA temporal (TCN +guard)   temporal head on vjepa2, +freezing guard
  DINOv2 temporal (TCN)          temporal head on dinov2, no freezing guard
  DINOv2 temporal (TCN +guard)   temporal head on dinov2, +freezing guard
  per-frame (<enc>)              the best per-frame-probe anchor (lower mean wins)
  BOCPD-pc1                      label-free change point, PC1 signal (changepoint_events)
  BOCPD-diffnorm                 label-free change point, diffnorm signal
  BEAST                          label-free offline change point
  RIce-Net                       the threshold baseline, IF results/ricenet_events.csv exists

Inputs (results/):
  phase3_timing_<enc>_<head>[_guard].csv   head/guard-tagged temporal+per-frame rows
                                           (the un-tagged phase3_timing_<enc>.csv files
                                           are STALE pre-grid runs and are ignored)
  changepoint_events.csv                   (label-free events)
  reference_events.csv                     (usgs_ice_flag truth)
  ricenet_events.csv                       (OPTIONAL; handled gracefully)

For changepoint/ricenet event rows the timing error is computed here against the
usgs_ice_flag reference via evaluation.metrics.timing_error_hours.

GATE B: compares V-JEPA-temporal vs DINOv2-temporal mean timing error at matched
configs (TCN+guard and TCN no-guard) and prints a recommendation, flagging that the
V-JEPA win is guard-dependent. The human makes the call; this lays out the comparison.

Output: results/phase4_h2_table.csv

Usage:
  python -m fm_ice.evaluation.phase4_table
  python -m fm_ice.evaluation.phase4_table --results results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.evaluation.metrics import timing_error_hours

REF = "usgs_ice_flag"   # the solid reference; stage_breakpoint is a rough cross-check

# The H2 comparison is fixed to the TCN head family (the Phase-3 winner) so every
# knob is identical across encoders; guard on/off is shown side by side.
HEAD = "tcn"


# --------------------------------------------------------------------------- #
# Held-out winters, discovered from the data (no hardcoded winter list).
# --------------------------------------------------------------------------- #
def discover_winters(results_dir: Path, station: str) -> list[str]:
    """Sorted held-out winters for `station`, from the tagged phase3 timing CSVs
    intersected with the reference events. Falls back to the reference set."""
    timing = set()
    for f in results_dir.glob(f"phase3_timing_*_*.csv"):
        df = pd.read_csv(f, usecols=lambda c: c in ("station", "test_winter"))
        timing |= set(str(w) for w in df[df["station"] == station]["test_winter"].unique())
    refs = set()
    rf = results_dir / "reference_events.csv"
    if rf.exists():
        rdf = pd.read_csv(rf)
        refs = set(str(w) for w in rdf[rdf["station"] == station]["winter"].unique())
    winters = (timing & refs) if (timing and refs) else (timing or refs)
    return sorted(winters)


# --------------------------------------------------------------------------- #
# Reference events keyed by (station, winter).
# --------------------------------------------------------------------------- #
def load_references(results_dir: Path) -> dict:
    f = results_dir / "reference_events.csv"
    if not f.exists():
        raise SystemExit(f"missing {f} -- run fm_ice.evaluation.reference_events first")
    df = pd.read_csv(f)
    df = df[df["source"] == REF]
    refs = {}
    for _, r in df.iterrows():
        refs[(r["station"], str(r["winter"]))] = {
            "onset": pd.Timestamp(r["onset_utc"]) if pd.notna(r["onset_utc"]) else None,
            "breakup": pd.Timestamp(r["breakup_utc"]) if pd.notna(r["breakup_utc"]) else None,
        }
    return refs


def _err_row(method: str, per_winter: dict, winters: list[str]) -> dict:
    """Assemble one table row from {winter: {'onset':h,'breakup':h}}."""
    row = {"method": method}
    errs = []
    for w in winters:
        e = per_winter.get(w, {})
        on = e.get("onset", np.nan)
        br = e.get("breakup", np.nan)
        row[f"{w} onset"] = on
        row[f"{w} breakup"] = br
        for v in (on, br):
            if not (v is None or (isinstance(v, float) and np.isnan(v))):
                errs.append(v)
    row["mean_err_h"] = round(float(np.mean(errs)), 1) if errs else np.nan
    return row


# --------------------------------------------------------------------------- #
# Phase-3 temporal + per-frame rows (errors are precomputed in those CSVs).
# --------------------------------------------------------------------------- #
def _load_tagged_timing(results_dir: Path) -> pd.DataFrame:
    """Concat the head/guard-tagged phase3 timing CSVs (the `_*` glob excludes the
    stale un-tagged phase3_timing_<enc>.csv files). Mirrors phase3_report.load_all."""
    files = sorted(results_dir.glob("phase3_timing_*_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "temp_guard" not in df:
        df["temp_guard"] = False
    return df[df["reference"] == REF].copy()


def _per_winter(g: pd.DataFrame) -> dict:
    return {str(r["test_winter"]): {"onset": r["onset_err_h"],
                                    "breakup": r["breakup_err_h"]}
            for _, r in g.iterrows()}


def _phase3_rows(results_dir: Path, winters: list[str]) -> list[dict]:
    rows = []
    df = _load_tagged_timing(results_dir)
    if df.empty:
        return rows

    label = {"vjepa2": "V-JEPA", "dinov2": "DINOv2"}

    # temporal head (TCN family): guard-off then guard-on row per encoder, so the
    # human can see how much of any V-JEPA edge rests on the freezing guard.
    for enc in ("vjepa2", "dinov2"):
        for guard in (False, True):
            g = df[(df["encoder"] == enc) & (df["head"] == HEAD)
                   & (df["model"] == "temporal_head")
                   & (df["temp_guard"] == guard)]
            if g.empty:
                continue
            tag = f" +guard" if guard else ""
            rows.append(_err_row(f"{label[enc]} temporal ({HEAD.upper()}{tag})",
                                 _per_winter(g), winters))

    # per-frame anchor (TCN, no guard): pick the encoder with the lower mean.
    best = None
    for enc in ("vjepa2", "dinov2"):
        g = df[(df["encoder"] == enc) & (df["head"] == HEAD)
               & (df["model"] == "perframe_probe") & (~df["temp_guard"])]
        if g.empty:
            continue
        cand = _err_row(f"per-frame ({label[enc]})", _per_winter(g), winters)
        if best is None or (not np.isnan(cand["mean_err_h"])
                            and cand["mean_err_h"] < best["mean_err_h"]):
            best = cand
    if best is not None:
        rows.append(best)
    return rows


# --------------------------------------------------------------------------- #
# Event-CSV rows (changepoint, ricenet): compute timing error here.
# --------------------------------------------------------------------------- #
def _event_err(onset, breakup, ref: dict) -> dict:
    pred = {
        "onset": pd.Timestamp(onset) if pd.notna(onset) else None,
        "breakup": pd.Timestamp(breakup) if pd.notna(breakup) else None,
    }
    te = timing_error_hours(pred, ref)
    return {"onset": te.get("onset", np.nan), "breakup": te.get("breakup", np.nan)}


def _changepoint_rows(results_dir: Path, refs: dict, station: str,
                      winters: list[str]) -> list[dict]:
    f = results_dir / "changepoint_events.csv"
    if not f.exists():
        print(f"[phase4] note: {f.name} absent -- skipping change-point rows")
        return []
    df = pd.read_csv(f)
    df = df[df["station"] == station]
    rows = []
    # one table row per (method, signal); BEAST collapses signals into one label.
    seen: dict = {}
    for _, r in df.iterrows():
        method, signal = r["method"], r["signal"]
        label = "BEAST" if method == "beast" else f"BOCPD-{signal}"
        ref = refs.get((r["station"], str(r["winter"])))
        if ref is None:
            continue
        per = seen.setdefault(label, {})
        per[str(r["winter"])] = _event_err(r["onset_utc"], r["breakup_utc"], ref)
    for label, per in seen.items():
        rows.append(_err_row(label, per, winters))
    return rows


def _ricenet_rows(results_dir: Path, refs: dict, station: str,
                  winters: list[str]) -> list[dict]:
    f = results_dir / "ricenet_events.csv"
    if not f.exists():
        print(f"[phase4] note: {f.name} absent -- RIce-Net row omitted (deferred GPU run)")
        return []
    df = pd.read_csv(f)
    df = df[df["station"] == station]
    per = {}
    for _, r in df.iterrows():
        ref = refs.get((r["station"], str(r["winter"])))
        if ref is None:
            continue
        per[str(r["winter"])] = _event_err(r["onset_utc"], r["breakup_utc"], ref)
    return [_err_row("RIce-Net", per, winters)] if per else []


# --------------------------------------------------------------------------- #
def build_table(results_dir: Path, station: str, winters: list[str]) -> pd.DataFrame:
    refs = load_references(results_dir)
    rows = []
    rows += _phase3_rows(results_dir, winters)
    rows += _changepoint_rows(results_dir, refs, station, winters)
    rows += _ricenet_rows(results_dir, refs, station, winters)
    if not rows:
        raise SystemExit("no Phase-4 inputs found in results/ -- run phase 3 + "
                         "changepoint_events first")
    cols = ["method"]
    for w in winters:
        cols += [f"{w} onset", f"{w} breakup"]
    cols += ["mean_err_h"]
    return pd.DataFrame(rows)[cols]


def gate_b(table: pd.DataFrame) -> None:
    def mean_for(name: str):
        m = table[table["method"] == name]["mean_err_h"]
        return float(m.iloc[0]) if len(m) and not np.isnan(m.iloc[0]) else None

    H = HEAD.upper()
    print("\n================ GATE B (H2): V-JEPA vs DINOv2 ================")

    def compare(tag: str, suffix: str) -> None:
        v = mean_for(f"V-JEPA temporal ({H}{suffix})")
        d = mean_for(f"DINOv2 temporal ({H}{suffix})")
        if v is None or d is None:
            print(f"  [{tag}] cannot decide -- missing a temporal row.")
            return
        verdict = (f"V-JEPA beats DINOv2 by {d - v:.1f} h" if v < d
                   else f"V-JEPA ties/loses DINOv2 by {v - d:.1f} h")
        print(f"  [{tag}] V-JEPA {v:.1f} h vs DINOv2 {d:.1f} h  ->  {verdict}")

    compare("TCN +guard ", " +guard")
    compare("TCN no-guard", "")
    print("Note: the freezing guard is a domain prior applied identically to both "
          "encoders; it changes V-JEPA's onset but not DINOv2's, so any V-JEPA win is "
          "guard-dependent. Report both rows honestly. Human makes the call.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--station", default="cedarburg")
    args = ap.parse_args()
    results_dir = Path(args.results)

    winters = discover_winters(results_dir, args.station)
    if not winters:
        raise SystemExit(f"no held-out winters found for {args.station} in "
                         f"{results_dir} -- run phase 3 first")
    table = build_table(results_dir, args.station, winters)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n===== Phase 4 (H2) onset/breakup timing error in hours vs {REF} =====")
    print(f"({args.station} leave-one-winter-out; lower is better)\n")
    print(table.to_string(index=False))

    gate_b(table)

    out = results_dir / "phase4_h2_table.csv"
    table.to_csv(out, index=False)
    print(f"\n[phase4] wrote {out}")


if __name__ == "__main__":
    main()
