"""Reference onset/breakup events per station-winter.

These are the GROUND-TRUTH-side events that predictions (and the RIce-Net
baseline) are scored against by fm_ice.evaluation.metrics. The research plan
names two independent references; both are produced here:

  usgs_ice_flag    -- from the daily ice flag (download_ice_flags). Onset = start
                      of the first sustained ice run; breakup = end of the last
                      ice run. Coarse (1-day) but operationally authoritative.
  stage_breakpoint -- label-free change points in the gage-height series. Ice
                      onset backs water up (stage rises and decouples from
                      discharge); breakup releases it. We detect change points
                      with the project's own BOCPD (fm_ice.baselines.changepoint)
                      and pick the dominant early-winter and spring transitions.

Both event times are tz-aware UTC. A daily ice event is anchored at the START of
its local-standard-time day expressed in UTC (date 00:00 CST -> 06:00 UTC).

Output: results/reference_events.csv
  station, winter, source, onset_utc, breakup_utc, n_changepoints, notes

Usage:
  python -m fm_ice.evaluation.reference_events --station cedarburg --winter 2022-2023
  python -m fm_ice.evaluation.reference_events --all          # every paper winter
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.baselines.changepoint import bocpd_gaussian, changepoints_from_runlength
from fm_ice.config import load_yaml, winter_bounds


def _day_start_utc(d, offset_hours: int) -> pd.Timestamp:
    """Local-standard-time calendar date -> UTC timestamp at its 00:00 local."""
    return pd.Timestamp(d, tz="UTC") - pd.Timedelta(hours=offset_hours)


# --------------------------------------------------------------------------- #
# Reference 1: USGS daily ice flag.
# --------------------------------------------------------------------------- #
def ice_flag_events(ice: pd.DataFrame, offset_hours: int, min_run: int = 2) -> dict:
    """Onset = first day that begins a run of >= min_run consecutive ice days.
    Breakup = the day AFTER the last ice day (ice gone). Both as UTC timestamps.

    min_run rejects isolated one-day blips, matching the persistence idea in the
    RIce-Net flag rule."""
    s = ice.sort_index()
    # Reindex to a CONTIGUOUS daily calendar so array positions are calendar days.
    # The ice CSV has a row only for days present in the USGS DV payload; on a gap
    # (outage), consecutive rows are not consecutive days, which would let a
    # run-length-over-rows scan see false "sustained" runs and misdate breakup.
    full = pd.date_range(min(s.index), max(s.index), freq="D")
    flag = (s["ice_flag"].astype("Int64")
            .reindex([d.date() for d in full]).fillna(0).astype(int).to_numpy())
    dates = [d.date() for d in full]
    n = len(flag)

    onset = None
    for i in range(n):
        if i + min_run <= n and flag[i] == 1 and flag[i:i + min_run].all():
            onset = _day_start_utc(dates[i], offset_hours)
            break

    breakup = None
    last_ice = next((i for i in range(n - 1, -1, -1) if flag[i] == 1), None)
    if last_ice is not None:
        # The day AFTER the last ice day, by calendar date (not next table row).
        breakup = _day_start_utc(dates[last_ice], offset_hours) + pd.Timedelta(days=1)

    return {"onset": onset, "breakup": breakup}


# --------------------------------------------------------------------------- #
# Reference 2: stage (gage-height) change points.
# --------------------------------------------------------------------------- #
def _hourly_stage(stage_csv: Path) -> pd.Series:
    s = pd.read_csv(stage_csv)
    s["datetime"] = pd.to_datetime(s["datetime"], utc=True)
    s = s[s["param"].astype(str).str.contains("65")] if "param" in s else s
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    # USGS NWIS uses numeric -999999 sentinels for missing/ice-affected readings;
    # to_numeric does not catch those, so mask them before averaging.
    s.loc[s["value"] < -999990, "value"] = pd.NA
    s = s.dropna(subset=["value"]).set_index("datetime").sort_index()
    return s["value"].resample("h").mean().interpolate(limit=6)


def stage_breakpoint_events(stage_csv: Path, hazard_lambda: float,
                            offset_hours: int) -> dict:
    """Detect stage change points with BOCPD and pick onset/breakup.

    Heuristic, label-free, and explicitly a reference to be cross-checked against
    the ice flag -- not treated as truth. Onset = the dominant change point in
    Nov-Feb; breakup = the dominant change point in Mar-Apr."""
    stage = _hourly_stage(stage_csv).dropna()
    if len(stage) < 100:
        return {"onset": None, "breakup": None, "n_changepoints": 0}

    x = (stage.to_numpy() - stage.mean()) / (stage.std() + 1e-9)
    res = bocpd_gaussian(x, hazard_lambda=hazard_lambda)
    cps = changepoints_from_runlength(res["map_run_length"], min_drop=24, min_sep=72)
    cp_times = [stage.index[i] for i in cps if i < len(stage)]

    def jump(t):  # magnitude of the level shift at a change point (for ranking)
        i = stage.index.get_loc(t)
        a = stage.iloc[max(0, i - 24):i].mean()
        b = stage.iloc[i:i + 24].mean()
        return abs(b - a)

    onset_cands = [(jump(t), t) for t in cp_times if (t + pd.Timedelta(hours=offset_hours)).month in (11, 12, 1, 2)]
    breakup_cands = [(jump(t), t) for t in cp_times if (t + pd.Timedelta(hours=offset_hours)).month in (3, 4)]
    onset = max(onset_cands)[1] if onset_cands else None
    breakup = max(breakup_cands)[1] if breakup_cands else None
    return {"onset": onset, "breakup": breakup, "n_changepoints": len(cp_times)}


# --------------------------------------------------------------------------- #
def run(station: str, winter: str) -> list[dict]:
    cfg_s = load_yaml("stations.yaml")
    cfg_p = load_yaml("pipeline.yaml")
    d = cfg_s["defaults"]
    st = cfg_s["stations"][station]
    nwis = st["nwis_id"]
    offset = int(st.get("usgs_dv_utc_offset_hours", d["usgs_dv_utc_offset_hours"]))
    hazard = cfg_p["changepoint"]["bocpd"]["hazard_lambda"]

    raw = Path(cfg_p["paths"]["raw"])
    ice_csv = raw / "ice_flags" / station / f"{nwis}_{winter}_iceflag.csv"
    stage_csv = raw / "stage" / station / f"{nwis}_{winter}_iv.csv"

    rows = []
    if ice_csv.exists():
        ice = pd.read_csv(ice_csv)
        ice["date"] = pd.to_datetime(ice["date"]).dt.date
        ev = ice_flag_events(ice.set_index("date"), offset)
        n_ice = int(ice["ice_flag"].sum())
        rows.append({"station": station, "winter": winter, "source": "usgs_ice_flag",
                     "onset_utc": ev["onset"], "breakup_utc": ev["breakup"],
                     "n_changepoints": "", "notes": f"{n_ice} ice days"})
    if stage_csv.exists():
        ev = stage_breakpoint_events(stage_csv, hazard, offset)
        rows.append({"station": station, "winter": winter, "source": "stage_breakpoint",
                     "onset_utc": ev["onset"], "breakup_utc": ev["breakup"],
                     "n_changepoints": ev.get("n_changepoints", ""),
                     "notes": f"BOCPD hazard={hazard}h"})

    for r in rows:
        print(f"[ref] {station} {winter} {r['source']:16s} "
              f"onset={r['onset_utc']}  breakup={r['breakup_utc']}  {r['notes']}")
    return rows


def write_results(all_rows: list[dict], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "reference_events.csv"
    cols = ["station", "winter", "source", "onset_utc", "breakup_utc", "n_changepoints", "notes"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[ref] wrote {len(all_rows)} rows -> {out}")
    return out


# The paper split: train Cedarburg, transfer-test Bismarck.
PAPER_WINTERS = [("cedarburg", "2022-2023"), ("cedarburg", "2023-2024"),
                 ("bismarck", "2024-2025")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station")
    ap.add_argument("--winter")
    ap.add_argument("--all", action="store_true", help="all paper station-winters")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    targets = PAPER_WINTERS if args.all else [(args.station, args.winter)]
    if any(s is None or w is None for s, w in targets):
        raise SystemExit("Provide --station and --winter, or --all.")
    all_rows = []
    for s, w in targets:
        all_rows.extend(run(s, w))
    write_results(all_rows, Path(args.results))


if __name__ == "__main__":
    main()
