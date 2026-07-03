"""Guard audit: re-read onset/breakup from the EXISTING per-clip probability
dumps under three guard variants, without retraining anything.

Why this exists (docs/FM_ice_plan_v2.md, addendum item 2): Gate B flips on the
freezing guard, so the guard must be auditable. The head's probabilities are
guard-invariant (the guard only filters sustained runs at event-reading time;
verified: prob columns are byte-identical between _guard and no-guard dumps),
so every guard variant can be re-scored from results/phase3_pred_*.csv alone.

Guard variants per row:
  none     no guard (primary evidence row).
  meanair  LEGACY mean-air-temperature guard (onset_freeze_guard_c), exactly the
           published Phase-3/5 behavior, including its flaw of gating breakup.
  afdd     onset-only AFDD guard: onset can fire only after accumulated freezing
           degree-days exceed tau = afdd_guard_frac * min(AFDD at reference
           onset) over CALIBRATION winters. Calibration is train-winters-only
           and respects leave-one-winter-out: scoring cedarburg winter W uses
           only the OTHER cedarburg winter(s); scoring the transfer station
           (bismarck) uses all cedarburg winters. The scored winter's own
           reference NEVER enters its tau.

Output: results/reextract/timing.csv (a subdirectory, so the phase4 glob over
phase3_timing_*.csv can never pick up these rows by accident).

Usage:
  python -m fm_ice.evaluation.reextract_events                # everything, tcn
  python -m fm_ice.evaluation.reextract_events --encoder vjepa2 --head tcn
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from fm_ice.config import load_yaml
from fm_ice.data.degree_days import (afdd, calibrate_tau_afdd, load_daily_mean_tmpc,
                                     station_utc_offset_hours, value_at_times)
from fm_ice.models.events import min_run_steps_from_hours, read_events
from fm_ice.models.train import evaluate_prediction

GUARDS = ("none", "meanair", "afdd")
MODELS = ("temporal_head", "perframe_probe")

# no-guard pred dumps only: the _guard twins carry identical probabilities.
PRED_RE = re.compile(r"phase3_pred_(?P<enc>[a-z0-9]+)_(?P<station>[a-z_]+)_"
                     r"(?P<winter>\d{4}-\d{4})_(?P<head>[a-z]+)\.csv$")


def discover_preds(results_dir: Path, encoder: str | None, head: str) -> list[dict]:
    out = []
    for f in sorted(results_dir.glob("phase3_pred_*.csv")):
        m = PRED_RE.match(f.name)
        if not m:
            continue                       # _guard twins and stale tags
        d = m.groupdict()
        if d["head"] != head or (encoder and d["enc"] != encoder):
            continue
        out.append({**d, "path": f})
    return out


def calib_winters_for(station: str, winter: str, train_station: str,
                      train_winters: list[str]) -> list[tuple[str, str]]:
    """Fold-appropriate calibration set (train winters only, LOO respected)."""
    if station == train_station:
        return [(train_station, w) for w in train_winters if w != winter]
    return [(train_station, w) for w in train_winters]


def run(results_dir: Path, encoder: str | None, head: str,
        train_station: str = "cedarburg") -> pd.DataFrame:
    cfg = load_yaml("pipeline.yaml")
    ev_cfg = cfg["events"]
    threshold = ev_cfg["ice_state_threshold"]
    min_run = min_run_steps_from_hours(ev_cfg["min_event_separation_hours"],
                                       cfg["clip"]["stride_hours"])
    guard_c = ev_cfg.get("onset_freeze_guard_c", 0.0)
    frac = ev_cfg.get("afdd_guard_frac", 0.5)

    preds = discover_preds(results_dir, encoder, head)
    if not preds:
        raise SystemExit(f"no phase3_pred_*_{head}.csv dumps in {results_dir}")
    train_winters = sorted({p["winter"] for p in preds if p["station"] == train_station})

    rows = []
    for p in preds:
        df = pd.read_csv(p["path"])
        times = pd.to_datetime(df["t_start_utc"], utc=True)
        air = df["air_tmpc"].to_numpy(dtype=float)

        calib = calib_winters_for(p["station"], p["winter"], train_station, train_winters)
        tau = calibrate_tau_afdd(calib, frac=frac, results_dir=results_dir)
        daily = load_daily_mean_tmpc(p["station"], p["winter"])
        clip_afdd = value_at_times(afdd(daily), times,
                                   station_utc_offset_hours(p["station"]))

        for model in MODELS:
            prob = df[f"prob_{model}"].to_numpy(dtype=float)
            for guard in GUARDS:
                kw = {}
                if guard == "meanair":
                    kw = {"air": air, "max_run_mean_air": guard_c}
                elif guard == "afdd":
                    kw = {"afdd": clip_afdd, "min_onset_afdd": tau}
                ev = read_events(times, prob, threshold=threshold,
                                 min_run_steps=min_run, **kw)
                for r in evaluate_prediction(ev, p["station"], p["winter"], results_dir):
                    rows.append({"encoder": p["enc"], "head": head, "model": model,
                                 "guard": guard, "station": p["station"],
                                 "test_winter": p["winter"],
                                 "pred_onset": ev["onset"], "pred_breakup": ev["breakup"],
                                 **r,
                                 "tau_afdd": round(tau, 1) if guard == "afdd" else None,
                                 "calib_winters": "+".join(w for _, w in calib)
                                                  if guard == "afdd" else None})
    return pd.DataFrame(rows)


def _report(df: pd.DataFrame) -> None:
    view = df[(df["reference"] == "usgs_ice_flag") & (df["model"] == "temporal_head")]
    print("\n===== guard audit: temporal head vs usgs_ice_flag (hours; onset | breakup) =====")
    hdr = f"{'encoder':8s} {'station':10s} {'winter':10s}" + \
          "".join(f" {g:>17s}" for g in GUARDS)
    print(hdr); print("-" * len(hdr))
    for (enc, st, w), g in view.groupby(["encoder", "station", "test_winter"]):
        cells = []
        for guard in GUARDS:
            r = g[g["guard"] == guard]
            if r.empty:
                cells.append(f"{'-':>17s}"); continue
            oe, be = r.iloc[0]["onset_err_h"], r.iloc[0]["breakup_err_h"]
            fmt = lambda v: f"{v:.0f}" if pd.notna(v) else "miss"
            cells.append(f"{fmt(oe):>8s}|{fmt(be):<8s}")
        print(f"{enc:8s} {st:10s} {w:10s}" + " ".join(cells))
    print("\nguard=none is the primary evidence row; meanair is the legacy "
          "(published) guard; afdd is the audited onset-only replacement.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default=None, choices=["vjepa2", "dinov2"],
                    help="default: all encoders found")
    ap.add_argument("--head", default="tcn", choices=["tcn", "transformer"])
    ap.add_argument("--train-station", default="cedarburg")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    results_dir = Path(args.results)
    df = run(results_dir, args.encoder, args.head, args.train_station)
    out_dir = results_dir / "reextract"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "timing.csv"
    df.to_csv(out, index=False)
    _report(df)
    print(f"\n[reextract] wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
