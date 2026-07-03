"""Read onset/breakup events from a predicted per-step ice-state sequence.

The temporal head emits one ice-state probability per clip-step (clips are
ordered in time, `stride_hours` apart). This module turns that probability
sequence into two event timestamps -- onset and breakup -- so they can be scored
against the references in `fm_ice.evaluation.reference_events` with the same
metric (`fm_ice.evaluation.metrics.timing_error_hours`).

Event definition (deliberately mirrors the USGS-flag reference so the comparison
is apples-to-apples):
  onset    = the start time of the FIRST sustained ice run.
  breakup  = the start time of the step AFTER the LAST sustained ice run
             (i.e. the moment ice is finally gone), matching "the day after the
             last ice day" in reference_events.ice_flag_events.

A "sustained" run is a maximal run of ice steps at least `min_run_steps` long.
Requiring sustain mirrors the reference's `min_run` gate and kills flicker:
isolated bright frames misread as ice cannot create a phantom onset, and a
trailing one-step blip cannot push breakup late. Interior short thaw gaps do not
matter -- onset/breakup span from the first to the last sustained run, so a brief
mid-winter thaw between two sustained runs is correctly bridged. This is the
event-level analogue of the MS-TCN smoothing loss the head trains with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _runs(state: np.ndarray):
    """Yield (start, end_exclusive, value) for each maximal constant run."""
    n = len(state)
    i = 0
    while i < n:
        j = i + 1
        while j < n and state[j] == state[i]:
            j += 1
        yield i, j, int(state[i])
        i = j


def sustained_ice_runs(state: np.ndarray, min_run_steps: int):
    """Return (start, end_exclusive) of every ice run at least min_run_steps long."""
    return [(i, j) for i, j, v in _runs(np.asarray(state, dtype=int))
            if v == 1 and (j - i) >= max(1, min_run_steps)]


def read_events(times, prob, threshold: float = 0.5, min_run_steps: int = 1,
                air=None, max_run_mean_air=None,
                afdd=None, min_onset_afdd=None) -> dict:
    """Onset/breakup timestamps from a per-step ice probability sequence.

    Parameters
    ----------
    times : sequence of timestamps in chronological order (one per step).
    prob  : per-step ice probability, same length and order as `times`.
    threshold : ice if prob >= threshold.
    min_run_steps : minimum sustained-run length (in steps) for an event to count.
    air : optional per-step air temperature (deg C), same order as `times`.
    max_run_mean_air : LEGACY freezing guard, kept for comparison rows. When set
        together with `air`, a sustained ice run only counts if its MEAN air
        temperature is at or below this value. Two documented flaws (see
        docs/FM_ice_plan_v2.md addendum 2): it gates BREAKUP runs too, and its
        inclusion was decided with test folds already scored. Superseded by the
        AFDD guard below; do not combine the two.
    afdd : optional per-step accumulated freezing degree-days (degC*day), same
        order as `times` (see fm_ice.data.degree_days.value_at_times).
    min_onset_afdd : AFDD onset guard, calibrated on TRAIN winters only
        (degree_days.calibrate_tau_afdd). Onset arms at the first step with
        afdd >= min_onset_afdd; a sustained run whose END precedes arming cannot
        be the onset (warm-season false positive), but a run that STRADDLES
        arming keeps its true start (calibration is deliberately lenient, so a
        genuine onset just before arming is not delayed). ONSET-ONLY: breakup is
        read from the unguarded run list -- thaw physics does not care how much
        freezing accumulated before.

    Returns {'onset': Timestamp|None, 'breakup': Timestamp|None,
             'state': np.ndarray, 'n_ice_steps': int}. onset/breakup are None
    when no sustained ice run exists (a winter the head calls open all season).
    With the AFDD guard, onset can be None while breakup is not (every run ended
    before arming: the guard calls them all warm-season artifacts).
    """
    t = pd.to_datetime(pd.Series(list(times)), utc=True).to_numpy()
    p = np.asarray(prob, dtype=float)
    if len(t) != len(p):
        raise ValueError(f"times ({len(t)}) and prob ({len(p)}) length mismatch")

    state = (p >= threshold).astype(int)
    runs = sustained_ice_runs(state, min_run_steps)
    if air is not None and max_run_mean_air is not None:
        a = np.asarray(air, dtype=float)
        runs = [(i, j) for (i, j) in runs if np.nanmean(a[i:j]) <= max_run_mean_air]
    if not runs:
        return {"onset": None, "breakup": None, "state": state, "n_ice_steps": 0}

    onset_runs = runs
    if afdd is not None and min_onset_afdd is not None:
        f = np.asarray(afdd, dtype=float)
        if len(f) != len(p):
            raise ValueError(f"afdd ({len(f)}) and prob ({len(p)}) length mismatch")
        armed = np.flatnonzero(f >= min_onset_afdd)
        t_arm = int(armed[0]) if len(armed) else len(p)   # never armed -> no onset
        onset_runs = [(i, j) for (i, j) in runs if j > t_arm]

    onset = pd.Timestamp(t[onset_runs[0][0]]) if onset_runs else None
    last_end = runs[-1][1]               # end-exclusive of the last sustained run
    # breakup = first step after the last sustained ice run; None if that run
    # reaches the end of the window (still frozen when observation stops).
    breakup = pd.Timestamp(t[last_end]) if last_end < len(t) else None
    n_ice = int(sum(j - i for i, j in runs))
    return {"onset": onset, "breakup": breakup,
            "state": state, "n_ice_steps": n_ice}


def min_run_steps_from_hours(min_hours: float, stride_hours: float) -> int:
    """Convert a sustained-event duration in hours to a run length in steps."""
    return max(1, int(round(min_hours / stride_hours)))


if __name__ == "__main__":
    # Self-test, no data needed.
    times = pd.date_range("2025-01-01", periods=20, freq="4h", tz="UTC")
    # flicker at 2, a one-step gap at 9, sustained ice 5..12, trailing blip at 18.
    prob = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                    dtype=float)
    ev = read_events(times, prob, threshold=0.5, min_run_steps=3)
    print("onset:", ev["onset"], " breakup:", ev["breakup"], " n_ice:", ev["n_ice_steps"])
    assert ev["onset"] == times[5], ev["onset"]        # blip at idx 2 not sustained
    assert ev["breakup"] == times[13], ev["breakup"]   # last sustained run 10..12 ends
    # all-open winter -> no events
    ev0 = read_events(times, np.zeros(20), min_run_steps=3)
    assert ev0["onset"] is None and ev0["breakup"] is None

    # freezing guard: a warm sustained ice run (5..8) is rejected, a cold one kept.
    air = np.full(20, -5.0); air[5:9] = 8.0          # run 5..8 is warm
    evg = read_events(times, prob, threshold=0.5, min_run_steps=3,
                      air=air, max_run_mean_air=0.0)
    assert evg["onset"] == times[10], evg["onset"]   # warm first run dropped

    # AFDD guard, onset-only. Runs (min 3 steps): 5..8 and 10..12.
    # (a) run entirely before arming is skipped; breakup is untouched.
    afdd_a = np.concatenate([np.zeros(10), np.linspace(10, 30, 10)])  # arms at idx 10
    eva = read_events(times, prob, threshold=0.5, min_run_steps=3,
                      afdd=afdd_a, min_onset_afdd=10.0)
    assert eva["onset"] == times[10], eva["onset"]   # first run (5..8) ends pre-arm
    assert eva["breakup"] == times[13], eva["breakup"]  # breakup NOT gated
    # (b) a run STRADDLING arming keeps its true start (lenient by design).
    afdd_b = np.concatenate([np.zeros(7), np.linspace(10, 30, 13)])   # arms at idx 7
    evb = read_events(times, prob, threshold=0.5, min_run_steps=3,
                      afdd=afdd_b, min_onset_afdd=10.0)
    assert evb["onset"] == times[5], evb["onset"]    # run 5..8 straddles arming
    # (c) never armed -> onset None, breakup still read from the runs.
    evc = read_events(times, prob, threshold=0.5, min_run_steps=3,
                      afdd=np.zeros(20), min_onset_afdd=10.0)
    assert evc["onset"] is None and evc["breakup"] == times[13]
    print("events self-test OK")
