"""Per-clip ice-affected labels from the USGS daily ice flag.

The temporal head learns a per-clip ice-state sequence; this module produces the
supervision for it. The label source is the USGS daily ice flag downloaded by
``download_ice_flags`` (the cold-season ``e`` / literal ``Ice`` qualifier on the
daily-mean discharge record). See that module for the qualifier semantics.

Time-zone subtlety (the easy bug here): USGS daily values are computed on the
LOCAL STANDARD-TIME day, and both stations report ``tz_cd=CST`` (UTC-6, no DST).
Clip timestamps are UTC. To map a clip to its ice day we shift the clip time by
``usgs_dv_utc_offset_hours`` (-6) and take the calendar date. Using naive UTC
dates would misassign every clip near local midnight.

A clip is 16 h wide, so it can straddle two local days. We therefore expose:
  * ``ice_flag``       -- the binary label = ice flag of the clip MIDPOINT's local
                          day. Deterministic; this is what the head trains on.
  * ``ice_day_frac``   -- fraction of the clip's hourly steps whose local day is
                          ice-flagged (soft label / boundary-clip diagnostic).
  * ``ice_explicit`` / ``ice_estimated`` -- the two qualifier components of the
                          midpoint day, kept so the literal-``Ice`` vs estimated-``e``
                          signal can be audited apart downstream.

Stage breakpoints are an INDEPENDENT onset/breakup reference, not a per-clip
label; they live in ``fm_ice.evaluation.reference_events``.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def load_ice_flags(nwis_id: str, winter: str, station: str, ice_root: Path) -> pd.DataFrame | None:
    """Load the per-day ice-flag CSV for a station-winter, indexed by date.

    Returns None if the file is missing (labels then come back as NA, not 0, so a
    missing download is never silently read as 'no ice')."""
    p = ice_root / station / f"{nwis_id}_{winter}_iceflag.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.set_index("date")


def _local_dates(t_start: pd.Timestamp, t_end: pd.Timestamp, offset_hours: int) -> list[date]:
    """Local-standard-time calendar dates touched by each hourly step in [start, end)."""
    hours = pd.date_range(t_start, t_end, freq="h", inclusive="left")
    shifted = hours + pd.Timedelta(hours=offset_hours)
    return [t.date() for t in shifted]


def _midpoint_local_date(t_start: pd.Timestamp, t_end: pd.Timestamp, offset_hours: int) -> date:
    mid = t_start + (t_end - t_start) / 2
    return (mid + pd.Timedelta(hours=offset_hours)).date()


def label_clips(clips: pd.DataFrame, ice: pd.DataFrame | None, offset_hours: int) -> pd.DataFrame:
    """Attach ice_flag / ice_day_frac / ice_explicit / ice_estimated to clips.

    clips must have tz-aware t_start_utc / t_end_utc. If ``ice`` is None the four
    columns are filled with pd.NA (unknown), never 0."""
    if ice is None or ice.empty:
        n = len(clips)
        clips["ice_flag"] = pd.array([pd.NA] * n, dtype="Int64")
        clips["ice_day_frac"] = pd.array([pd.NA] * n, dtype="Float64")
        clips["ice_explicit"] = pd.array([pd.NA] * n, dtype="Int64")
        clips["ice_estimated"] = pd.array([pd.NA] * n, dtype="Int64")
        return clips

    flag_map = ice["ice_flag"].to_dict()
    expl_map = ice["ice_explicit"].to_dict()
    est_map = ice["estimated"].to_dict()

    ice_flag, ice_frac, ice_expl, ice_est = [], [], [], []
    for _, c in clips.iterrows():
        mid_d = _midpoint_local_date(c["t_start_utc"], c["t_end_utc"], offset_hours)
        days = _local_dates(c["t_start_utc"], c["t_end_utc"], offset_hours)
        known = [flag_map[d] for d in days if d in flag_map]

        ice_flag.append(int(flag_map[mid_d]) if mid_d in flag_map else pd.NA)
        # Fraction over the FULL clip window: an unknown day's hours count as
        # not-ice in the denominator (matches the docstring) so boundary clips at
        # the season edge are not silently rescaled upward.
        ice_frac.append(float(sum(known) / len(days)) if days else pd.NA)
        ice_expl.append(int(expl_map[mid_d]) if mid_d in expl_map else pd.NA)
        ice_est.append(int(est_map[mid_d]) if mid_d in est_map else pd.NA)

    # Nullable extension dtypes so parquet preserves integer/float semantics and
    # keeps NA (unknown) distinct from 0 -- a bare int+pd.NA list becomes object
    # and round-trips through parquet as float64/NaN, losing the label type.
    clips["ice_flag"] = pd.array(ice_flag, dtype="Int64")
    clips["ice_day_frac"] = pd.array(ice_frac, dtype="Float64")
    clips["ice_explicit"] = pd.array(ice_expl, dtype="Int64")
    clips["ice_estimated"] = pd.array(ice_est, dtype="Int64")
    return clips
