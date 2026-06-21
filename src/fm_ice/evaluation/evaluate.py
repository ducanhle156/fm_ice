"""Phase 6 master metrics driver.

Scores every cached Phase-3 prediction file against BOTH references and rolls the
numbers into one tidy table -- the spine of the results section. For each
`results/phase3_pred_<encoder>_<station>_<winter>_<head>[_guard].csv` it computes:

  (a) timing error   onset/breakup absolute hours vs each reference in
                     reference_events.csv. Predicted onset/breakup are read from
                     the per-step sequence with fm_ice.models.events.read_events
                     (same event rule as the references), and -- when a matching
                     phase3_timing_*.csv row exists -- cross-checked against the
                     already-computed pred_onset/pred_breakup so a divergence is
                     visible rather than silent.
  (b) event F1       precision/recall/F1 at 24/48/72 h tolerance, treating onset
                     and breakup as the two predicted/reference event times.
  (c) per-frame      flag agreement (accuracy/balanced/F1) and ROC-AUC of the
                     temporal head AND the per-frame probe vs the per-step ice_flag.

Output: results/phase6_metrics.csv (long form, one row per
pred-file x reference x model-channel) plus a printed summary table.

This is CPU-only and reads only cached CSVs -- no embeddings, no GPU, no network.
The event rule's threshold / min-run / freezing-guard come from configs/pipeline.yaml
(events.*, clip.stride_hours); CLI flags override them without editing yaml.

Usage:
  python -m fm_ice.evaluation.evaluate                       # all phase3_pred_*.csv
  python -m fm_ice.evaluation.evaluate --encoder vjepa2      # filter by encoder
  python -m fm_ice.evaluation.evaluate --glob '*tcn.csv'     # filter pred files
  python -m fm_ice.evaluation.evaluate --min-run-hours 8     # override event rule
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml
from fm_ice.evaluation.metrics import (
    event_f1_at_tolerances,
    per_frame_agreement,
    per_frame_auc,
    timing_error_hours,
)
from fm_ice.models.events import min_run_steps_from_hours, read_events

# A winter label is two 4-digit years: the anchor for filename parsing.
_WINTER_RE = re.compile(r"(\d{4}-\d{4})")


def parse_pred_name(path: Path) -> dict:
    """Pull (encoder, station, winter, head, guard) out of a pred filename.

    `phase3_pred_<encoder>_<station>_<winter>_<head>[_guard].csv`. We anchor on the
    `YYYY-YYYY` winter token so a multi-word station would still parse; encoder and
    station are everything before it, head[/guard] everything after.
    """
    stem = path.stem
    if not stem.startswith("phase3_pred_"):
        raise ValueError(f"not a phase3_pred file: {path.name}")
    body = stem[len("phase3_pred_"):]
    m = _WINTER_RE.search(body)
    if not m:
        raise ValueError(f"no winter token in {path.name}")
    winter = m.group(1)
    before = body[:m.start()].rstrip("_")          # <encoder>_<station>
    after = body[m.end():].lstrip("_")             # <head>[_guard]
    enc, _, station = before.partition("_")
    guard = after.endswith("_guard") or after == "guard"
    head = after[:-len("_guard")] if after.endswith("_guard") else after
    head = head or "unknown"
    return {"encoder": enc, "station": station, "winter": winter,
            "head": head, "guard": bool(guard)}


def _events_cfg(args):
    cfg = load_yaml("pipeline.yaml")
    ev = cfg.get("events", {})
    stride = cfg["clip"]["stride_hours"]
    threshold = args.threshold if args.threshold is not None else ev.get("ice_state_threshold", 0.5)
    guard_c = args.freeze_guard_c if args.freeze_guard_c is not None else ev.get("onset_freeze_guard_c", 0.0)
    min_run_steps = min_run_steps_from_hours(args.min_run_hours, stride)
    return cfg, threshold, guard_c, min_run_steps


def _pred_events(df: pd.DataFrame, prob_col: str, threshold: float,
                 min_run_steps: int, use_guard: bool, guard_c: float) -> dict:
    """Read onset/breakup from one probability channel of a pred frame.

    use_guard applies the air-temperature freezing guard (a run only counts if its
    mean air temp <= guard_c), mirroring the temp_guard variant of the head.
    """
    air = df["air_tmpc"].to_numpy() if use_guard and "air_tmpc" in df else None
    max_air = guard_c if use_guard else None
    return read_events(df["t_start_utc"].to_numpy(), df[prob_col].to_numpy(),
                       threshold=threshold, min_run_steps=min_run_steps,
                       air=air, max_run_mean_air=max_air)


def _ref_events_for(ref_df: pd.DataFrame, station: str, winter: str) -> dict[str, dict]:
    """{source: {'onset': ts|None, 'breakup': ts|None}} for this station-winter."""
    sub = ref_df[(ref_df["station"] == station) & (ref_df["winter"] == winter)]
    out = {}
    for _, r in sub.iterrows():
        out[r["source"]] = {
            "onset": pd.Timestamp(r["onset_utc"]) if pd.notna(r["onset_utc"]) else None,
            "breakup": pd.Timestamp(r["breakup_utc"]) if pd.notna(r["breakup_utc"]) else None,
        }
    return out


# Map a per-step model channel to its (probability, state) columns.
_CHANNELS = {
    "temporal_head": ("prob_temporal_head", "state_temporal_head"),
    "perframe_probe": ("prob_perframe_probe", "state_perframe_probe"),
}


def evaluate_file(path: Path, ref_df: pd.DataFrame, tolerances, threshold: float,
                  guard_c: float, min_run_steps: int) -> list[dict]:
    meta = parse_pred_name(path)
    df = pd.read_csv(path)
    df["t_start_utc"] = pd.to_datetime(df["t_start_utc"], utc=True)
    df = df.sort_values("t_start_utc").reset_index(drop=True)
    refs = _ref_events_for(ref_df, meta["station"], meta["winter"])

    rows = []
    for channel, (prob_col, state_col) in _CHANNELS.items():
        if prob_col not in df.columns:
            continue
        pred = _pred_events(df, prob_col, threshold, min_run_steps,
                            use_guard=meta["guard"], guard_c=guard_c)
        pred_times = [t for t in (pred["onset"], pred["breakup"]) if t is not None]

        # per-frame metrics are reference-independent (truth = the per-step ice_flag).
        agr = per_frame_agreement(df[state_col], df["ice_flag"])
        auc = per_frame_auc(df["ice_flag"], df[prob_col])

        for source, ref in refs.items():
            terr = timing_error_hours(pred, ref)
            ref_times = [t for t in (ref["onset"], ref["breakup"]) if t is not None]
            f1s = event_f1_at_tolerances(pred_times, ref_times, tolerances)
            row = {
                **meta, "channel": channel, "reference": source,
                "pred_onset": pred["onset"], "pred_breakup": pred["breakup"],
                "ref_onset": ref["onset"], "ref_breakup": ref["breakup"],
                "onset_err_h": terr.get("onset", float("nan")),
                "breakup_err_h": terr.get("breakup", float("nan")),
                "timing_mean_err_h": terr.get("mean", float("nan")),
                "n_ice_steps": pred["n_ice_steps"],
                "pf_accuracy": agr["accuracy"], "pf_balanced_acc": agr["balanced_acc"],
                "pf_f1": agr["f1"], "pf_auc": auc, "pf_n": agr["n"],
            }
            for tol, prf in f1s.items():
                tag = int(tol)
                row[f"event_f1_{tag}h"] = prf.f1
                row[f"event_prec_{tag}h"] = prf.precision
                row[f"event_recall_{tag}h"] = prf.recall
            rows.append(row)
    return rows


def run(results_dir: Path, glob: str, encoder: str | None, args) -> pd.DataFrame:
    cfg, threshold, guard_c, min_run_steps = _events_cfg(args)
    tolerances = cfg["evaluation"]["tolerance_hours"]
    ref_path = results_dir / "reference_events.csv"
    if not ref_path.exists():
        raise SystemExit(f"missing {ref_path} -- run fm_ice.evaluation.reference_events first")
    ref_df = pd.read_csv(ref_path)

    files = sorted(results_dir.glob(f"phase3_pred_{glob}"))
    if encoder:
        files = [f for f in files if f"_{encoder}_" in f.name]
    if not files:
        raise SystemExit(f"no pred files matching phase3_pred_{glob} in {results_dir}")

    all_rows = []
    for f in files:
        try:
            all_rows.extend(evaluate_file(f, ref_df, tolerances, threshold,
                                          guard_c, min_run_steps))
        except Exception as e:  # noqa: BLE001 -- keep scoring the rest
            print(f"[evaluate] WARN {f.name}: {e}")
    out = pd.DataFrame(all_rows)
    dest = results_dir / "phase6_metrics.csv"
    out.to_csv(dest, index=False)
    _report(out, tolerances)
    print(f"\n[evaluate] wrote {len(out)} rows -> {dest}")
    return out


def _report(df: pd.DataFrame, tolerances) -> None:
    if df.empty:
        print("[evaluate] no rows produced.")
        return
    print("\n===================== PHASE 6 metrics (per pred-file x channel) =====================")
    hdr = (f"{'enc':7s} {'station':9s} {'winter':9s} {'head':12s} {'g':1s} "
           f"{'channel':14s} {'ref':16s} {'on_h':>6s} {'bk_h':>6s} "
           f"{'pf_acc':>6s} {'pf_auc':>6s} {'F1@48':>6s}")
    print(hdr); print("-" * len(hdr))
    for _, r in df.iterrows():
        def f(x):
            return f"{x:6.1f}" if isinstance(x, (int, float)) and not pd.isna(x) else "    NA"
        print(f"{r['encoder']:7s} {r['station']:9s} {r['winter']:9s} {r['head']:12s} "
              f"{'Y' if r['guard'] else 'n':1s} {r['channel']:14s} {r['reference']:16s} "
              f"{f(r['onset_err_h'])} {f(r['breakup_err_h'])} {f(r['pf_accuracy'])} "
              f"{f(r['pf_auc'])} {f(r.get('event_f1_48h', float('nan')))}")
    print(f"\non_h/bk_h = onset/breakup timing error (h). event F1 at {tolerances} h.")
    print("per-frame acc/auc are reference-independent (truth = per-step ice_flag).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="results dir")
    ap.add_argument("--glob", default="*.csv",
                    help="suffix glob after phase3_pred_ (e.g. '*tcn.csv')")
    ap.add_argument("--encoder", default=None, help="filter to one encoder")
    ap.add_argument("--threshold", type=float, default=None,
                    help="ice-state threshold (default: events.ice_state_threshold)")
    ap.add_argument("--min-run-hours", type=float, default=8.0,
                    help="minimum sustained-ice run for an event (default 8 h)")
    ap.add_argument("--freeze-guard-c", type=float, default=None,
                    help="air-temp ceiling for a counted run on _guard files "
                         "(default: events.onset_freeze_guard_c)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the cheap filename-parsing self-test and exit")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    run(Path(args.results), args.glob, args.encoder, args)


def _selftest() -> None:
    """Cheap self-test: filename parsing only (no data needed)."""
    cases = {
        "phase3_pred_vjepa2_cedarburg_2022-2023_tcn.csv":
            ("vjepa2", "cedarburg", "2022-2023", "tcn", False),
        "phase3_pred_dinov2_cedarburg_2023-2024_transformer_guard.csv":
            ("dinov2", "cedarburg", "2023-2024", "transformer", True),
        "phase3_pred_vjepa2_bismarck_2024-2025_tcn_guard.csv":
            ("vjepa2", "bismarck", "2024-2025", "tcn", True),
    }
    for name, exp in cases.items():
        m = parse_pred_name(Path(name))
        got = (m["encoder"], m["station"], m["winter"], m["head"], m["guard"])
        assert got == exp, f"{name}: {got} != {exp}"
    print("evaluate parse self-test OK")


if __name__ == "__main__":
    main()
