"""RIce-Net threshold baseline: the anchor this project must beat.

Pipeline (Ayyad et al. 2025), reproduced as the baseline:
  1. Per image: segment ice vs water over the river mask (PAN decoder, ResNet50
     encoder). Use the published RIce-Net weights from HydroShare.
  2. Ice coverage IC(t) = ice_pixels / river_pixels * 100.
  3. Flag rule: flag on at t if IC(t') > tau for all t' in [t-8h, t].
       tau = 15 for onset, tau = 20 for breakup.
  4. Onset = first flag-on transition; breakup = corresponding flag-off in spring.

Only step 3-4 are fully implemented here (the rule is simple and exact). Step 1-2
need the trained segmentation model and the per-site river mask:
  - weights + masks: HydroShare dataset linked from https://github.com/m-ayyad/RIce-Net
  - model: segmentation_models_pytorch.PAN(encoder_name='resnet50')

The point of this module is to produce baseline onset/breakup dates that
fm_ice.evaluation scores against the references. Beating these dates on timing
error is the headline result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ONSET_TAU = 15.0
BREAKUP_TAU = 20.0
PERSIST_HOURS = 8


def segment_ice_coverage(image_paths, river_mask, model) -> pd.Series:
    """TODO: run the PAN segmenter over each image, restrict to river_mask, and
    return IC(t) in percent indexed by timestamp.

    Implement with segmentation_models_pytorch + RIce-Net weights. Keep this the
    ONLY place the segmentation model is used, so the rest of the baseline is
    model-agnostic and testable on synthetic coverage series.
    """
    raise NotImplementedError("Load RIce-Net weights + river mask; see module docstring.")


def threshold_persistence_flag(coverage: pd.Series, tau: float, persist_hours: int = PERSIST_HOURS) -> pd.Series:
    """Flag is on at t if coverage > tau for the whole trailing persist_hours window.

    coverage: float Series indexed by tz-aware UTC datetimes (hourly or finer).
    Returns a boolean Series aligned to coverage.index.
    """
    cov = coverage.sort_index()
    above = cov > tau
    flags = []
    for t in cov.index:
        window = above.loc[t - pd.Timedelta(hours=persist_hours): t]
        flags.append(bool(window.all()) and len(window) > 0)
    return pd.Series(flags, index=cov.index)


def onset_breakup_dates(coverage: pd.Series) -> dict[str, object]:
    """Read onset and breakup from the coverage series using the RIce-Net rule.

    Onset: first rising edge of the onset-threshold flag.
    Breakup: first falling edge of the breakup-threshold flag after onset.
    """
    onset_flag = threshold_persistence_flag(coverage, ONSET_TAU)
    breakup_flag = threshold_persistence_flag(coverage, BREAKUP_TAU)

    onset = None
    rises = onset_flag & ~onset_flag.shift(1, fill_value=False)
    if rises.any():
        onset = rises[rises].index[0]

    breakup = None
    if onset is not None:
        after = breakup_flag.loc[onset:]
        falls = ~after & after.shift(1, fill_value=False)
        if falls.any():
            breakup = falls[falls].index[0]
    return {"onset": onset, "breakup": breakup}


if __name__ == "__main__":
    # Self-test on a synthetic coverage series: ramps up in Jan, down in Mar.
    idx = pd.date_range("2025-12-01", "2026-04-01", freq="h", tz="UTC")
    cov = pd.Series(np.zeros(len(idx)), index=idx)
    cov.loc["2026-01-10":"2026-03-05"] = 60.0
    print(onset_breakup_dates(cov))
