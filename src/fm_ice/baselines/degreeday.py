"""Degree-day baseline (plan v2 Sec. 9): onset/breakup from accumulated
freezing / thawing degree-days alone. CPU only, no images, no embeddings.

This is the thermal control row: degree-day methods are the classical
operational approach, and their failure to time mechanical events is part of
the paper's argument. Expect it to be genuinely weak at breakup (measured ATDD
at reference breakup spans ~2 to ~56 degC*day across winters) -- that is its
job.

Rules (thresholds calibrated on TRAIN winters only, then frozen; the same
leave-one-winter-out discipline as the temporal head):
  onset    local midnight of the first day with AFDD >= tau_f, where tau_f is
           the MEAN over calibration winters of AFDD at the reference onset.
           (Mean, not the guard's halved-min: this row PREDICTS onset, it does
           not merely arm a detector.)
  breakup  anchor at the cumulatively coldest day at/after the predicted onset
           (the not_before constraint is mandatory: on a warm winter the
           unconstrained argmax collapses to Nov 1), accumulate thawing
           degree-days from there, fire at the first day with ATDD >= tau_t.
           tau_t = MEAN over calibration winters of ATDD at the reference
           breakup, anchored the same way but at the REFERENCE onset (known on
           train winters). Never crossed by May 15 -> None (metrics skip it).

Calibration folds (mirrors models/train.py + reextract_events):
  cedarburg winter W  -> calibrate on the other cedarburg winter(s)
  any other station   -> calibrate on all cedarburg winters

Output: results/degreeday_events.csv, consumed by evaluation.phase4_table.

Usage:
  python -m fm_ice.baselines.degreeday --all
  python -m fm_ice.baselines.degreeday --station bismarck --winter 2024-2025
  python -m fm_ice.baselines.degreeday --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml, station_utc_offset_hours
from fm_ice.data.degree_days import (afdd, atdd, cold_season_anchor,
                                     load_daily_mean_tmpc, value_at_event)

TRAIN_STATION = "cedarburg"


# --------------------------------------------------------------------------- #
# Pure prediction rule (self-testable on synthetic daily series)
# --------------------------------------------------------------------------- #
def _local_midnight_utc(day: pd.Timestamp, utc_offset_hours: int) -> pd.Timestamp:
    """Naive local-standard date -> the UTC timestamp of its local midnight
    (matches the reference-event convention, e.g. 06:00 UTC for CST)."""
    return (pd.Timestamp(day) - pd.Timedelta(hours=utc_offset_hours)).tz_localize("UTC")


def predict_events(daily: pd.Series, tau_f: float, tau_t: float,
                   utc_offset_hours: int) -> dict:
    """Onset/breakup from a daily-mean temperature series and frozen thresholds."""
    A = afdd(daily)
    crossed = A[A >= tau_f]
    if crossed.empty:
        return {"onset": None, "breakup": None, "anchor_day": None}
    onset_day = crossed.index[0]

    anchor = cold_season_anchor(daily, not_before=onset_day)
    T = atdd(daily, anchor)
    t_crossed = T[T >= tau_t]
    breakup_day = t_crossed.index[0] if not t_crossed.empty else None
    return {"onset": _local_midnight_utc(onset_day, utc_offset_hours),
            "breakup": (_local_midnight_utc(breakup_day, utc_offset_hours)
                        if breakup_day is not None else None),
            "anchor_day": anchor}


# --------------------------------------------------------------------------- #
# Calibration (train winters only)
# --------------------------------------------------------------------------- #
def _references(results_dir: Path) -> pd.DataFrame:
    f = results_dir / "reference_events.csv"
    if not f.exists():
        raise SystemExit(f"missing {f} -- run fm_ice.evaluation.reference_events first")
    df = pd.read_csv(f)
    return df[df["source"] == "usgs_ice_flag"]


def calibrate(calib: list[tuple[str, str]], results_dir: Path) -> tuple[float, float]:
    """(tau_f, tau_t) = mean AFDD at reference onset / mean ATDD at reference
    breakup over the calibration station-winters."""
    refs = _references(results_dir)
    fs, ts = [], []
    for station, winter in calib:
        row = refs[(refs["station"] == station) & (refs["winter"].astype(str) == winter)]
        if row.empty:
            raise SystemExit(f"no usgs_ice_flag reference for {station} {winter}")
        onset = pd.Timestamp(row.iloc[0]["onset_utc"])
        breakup = pd.Timestamp(row.iloc[0]["breakup_utc"])
        offset = station_utc_offset_hours(station)
        daily = load_daily_mean_tmpc(station, winter)
        fs.append(value_at_event(afdd(daily), onset, offset))
        # anchor for calibration at the (known) reference onset day
        onset_day = (onset + pd.Timedelta(hours=offset)).tz_localize(None).floor("D")
        anchor = cold_season_anchor(daily, not_before=onset_day)
        ts.append(value_at_event(atdd(daily, anchor), breakup, offset))
    return float(np.mean(fs)), float(np.mean(ts))


def calib_winters_for(station: str, winter: str, train_winters: list[str]) -> list[tuple[str, str]]:
    if station == TRAIN_STATION:
        return [(TRAIN_STATION, w) for w in train_winters if w != winter]
    return [(TRAIN_STATION, w) for w in train_winters]


# --------------------------------------------------------------------------- #
def run_one(station: str, winter: str, train_winters: list[str],
            results_dir: Path) -> dict:
    calib = calib_winters_for(station, winter, train_winters)
    if not calib:
        raise SystemExit(f"no calibration winters left for {station} {winter}")
    tau_f, tau_t = calibrate(calib, results_dir)
    daily = load_daily_mean_tmpc(station, winter)
    ev = predict_events(daily, tau_f, tau_t, station_utc_offset_hours(station))
    row = {"station": station, "winter": winter, "method": "degreeday",
           "onset_utc": ev["onset"], "breakup_utc": ev["breakup"],
           "tau_f": round(tau_f, 1), "tau_t": round(tau_t, 1),
           "anchor_day": (str(ev["anchor_day"].date())
                          if ev["anchor_day"] is not None else None),
           "calib_winters": "+".join(w for _, w in calib),
           "notes": "AFDD/ATDD thermal control; thresholds frozen on train winters"}
    print(f"[degreeday] {station} {winter}: onset={ev['onset']} "
          f"breakup={ev['breakup']} (tau_f={tau_f:.1f}, tau_t={tau_t:.1f}, "
          f"calib={row['calib_winters']})")
    return row


def discover_targets(results_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Station-winters with a usgs_ice_flag reference AND a temperature CSV."""
    refs = _references(results_dir)
    targets, train_winters = [], []
    from fm_ice.data.degree_days import station_temp_csv
    for _, r in refs.iterrows():
        st, w = r["station"], str(r["winter"])
        if not station_temp_csv(st, w).exists():
            print(f"[degreeday] skip {st} {w}: no temperature CSV")
            continue
        targets.append((st, w))
        if st == TRAIN_STATION:
            train_winters.append(w)
    return targets, sorted(train_winters)


def _selftest() -> None:
    days = pd.date_range("2024-11-01", "2025-05-15", freq="D")
    temp = np.full(len(days), 5.0)
    temp[(days >= "2024-12-01") & (days <= "2025-02-28")] = -10.0
    temp[days >= "2025-03-01"] = 8.0
    daily = pd.Series(temp, index=days)

    # tau_f = 50 -> AFDD crosses on Dec 5 (5 days x 10 deg); local midnight CST.
    ev = predict_events(daily, tau_f=50.0, tau_t=40.0, utc_offset_hours=-6)
    assert str(ev["onset"]) == "2024-12-05 06:00:00+00:00", ev["onset"]
    # anchor = last freezing day (Feb 28); ATDD crosses 40 on Mar 5 (5 x 8 deg).
    assert ev["anchor_day"] == pd.Timestamp("2025-02-28"), ev["anchor_day"]
    assert str(ev["breakup"]) == "2025-03-05 06:00:00+00:00", ev["breakup"]
    # never freezes enough -> no events at all.
    ev2 = predict_events(pd.Series(np.full(len(days), 4.0), index=days),
                         tau_f=50.0, tau_t=40.0, utc_offset_hours=-6)
    assert ev2["onset"] is None and ev2["breakup"] is None
    # tau_t never crossed -> onset fires, breakup None.
    cold_end = pd.Series(np.where(days < pd.Timestamp("2025-01-01"), -10.0, -1.0), index=days)
    ev3 = predict_events(cold_end, tau_f=50.0, tau_t=40.0, utc_offset_hours=-6)
    assert ev3["onset"] is not None and ev3["breakup"] is None
    print("degreeday self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="every station-winter with a reference + temps")
    ap.add_argument("--station")
    ap.add_argument("--winter")
    ap.add_argument("--results", default="results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    results_dir = Path(args.results)
    targets, train_winters = discover_targets(results_dir)
    if not args.all:
        if not (args.station and args.winter):
            raise SystemExit("provide --all or --station + --winter")
        targets = [(args.station, args.winter)]

    rows = [run_one(st, w, train_winters, results_dir) for st, w in targets]
    out = results_dir / "degreeday_events.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[degreeday] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
