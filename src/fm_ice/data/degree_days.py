"""Accumulated freezing / thawing degree-days (AFDD / ATDD) from the hourly
ASOS temperature CSVs.

Used by two consumers:
  - the AFDD onset guard (fm_ice.models.events.read_events, min_onset_afdd):
    onset can only fire after AFDD exceeds a threshold calibrated on TRAIN
    winters only (replaces the mean-air-temperature guard whose inclusion was
    decided with test folds visible -- see docs/FM_ice_plan_v2.md addendum 2).
  - the degree-day baseline (fm_ice.baselines.degreeday).

Conventions (must match reference_events.py):
  - Daily means are taken on the LOCAL STANDARD-TIME day. USGS daily values use
    a fixed standard-time offset (stations.yaml usgs_dv_utc_offset_hours,
    per-station override wins over the default), and reference events sit at
    06:00 UTC == local midnight CST.
  - AFDD(t) = cumulative sum over days of max(0, -daily_mean_tmpc), from the
    first day of the winter season window (Nov 1). Units: degC * day.
  - ATDD(t) = cumulative sum of max(0, +daily_mean_tmpc) from a spring anchor
    (the day AFDD peaks, i.e. the coldest point of the season cumulatively).

Missing days are interpolated BEFORE the cumsum: one NaN day would otherwise
poison every later value.

Self-test (no data or network):
  python -m fm_ice.data.degree_days
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml, winter_bounds


# --------------------------------------------------------------------------- #
# Station helpers
# --------------------------------------------------------------------------- #
def station_utc_offset_hours(station: str) -> int:
    """Per-station usgs_dv_utc_offset_hours, falling back to the default."""
    cfg = load_yaml("stations.yaml")
    st = cfg["stations"][station]
    return int(st.get("usgs_dv_utc_offset_hours",
                      cfg["defaults"]["usgs_dv_utc_offset_hours"]))


def station_temp_csv(station: str, winter: str, raw_dir: str | Path = "data/raw") -> Path:
    """Path of the hourly temperature CSV for a station-winter."""
    cfg = load_yaml("stations.yaml")
    asos = cfg["stations"][station]["asos_station"]
    return Path(raw_dir) / "temperature" / station / f"{asos}_{winter}_tmp.csv"


# --------------------------------------------------------------------------- #
# Daily series
# --------------------------------------------------------------------------- #
def daily_mean_from_hourly(hourly: pd.DataFrame, utc_offset_hours: int,
                           start: pd.Timestamp | None = None,
                           end: pd.Timestamp | None = None) -> pd.Series:
    """Daily mean tmpc on the local-standard day, gap-free.

    hourly needs columns datetime_utc + tmpc. The index of the returned series
    is the naive local-standard date (midnight timestamps); missing days inside
    [start, end] are linearly interpolated, edges back/forward-filled.
    """
    t = pd.to_datetime(hourly["datetime_utc"], utc=True)
    local_day = (t + pd.Timedelta(hours=utc_offset_hours)).dt.floor("D").dt.tz_localize(None)
    daily = (pd.Series(hourly["tmpc"].astype(float).to_numpy(), index=local_day)
             .groupby(level=0).mean())
    lo = start if start is not None else daily.index.min()
    hi = end if end is not None else daily.index.max()
    daily = daily.reindex(pd.date_range(lo, hi, freq="D"))
    n_missing = int(daily.isna().sum())
    if n_missing:
        print(f"[degree_days] interpolating {n_missing} missing day(s)")
        daily = daily.interpolate(limit_direction="both")
    return daily


def load_daily_mean_tmpc(station: str, winter: str,
                         raw_dir: str | Path = "data/raw") -> pd.Series:
    """Daily mean air temperature (degC) for one station-winter season window."""
    path = station_temp_csv(station, winter, raw_dir)
    if not path.exists():
        raise SystemExit(f"missing temperature CSV {path} -- run "
                         f"fm_ice.data.download_temperature first")
    hourly = pd.read_csv(path)
    st_cfg = load_yaml("stations.yaml")["defaults"]
    start, end = winter_bounds(winter, st_cfg["season_start_md"], st_cfg["season_end_md"])
    daily = daily_mean_from_hourly(hourly, station_utc_offset_hours(station),
                                   start=pd.Timestamp(start), end=pd.Timestamp(end))
    first_obs = pd.to_datetime(hourly["datetime_utc"], utc=True).min()
    if first_obs.tz_localize(None) > pd.Timestamp(start) + pd.Timedelta(days=1):
        print(f"[degree_days] WARNING: {station} {winter} temps start {first_obs:%Y-%m-%d}, "
              f"after the season start -- AFDD is underestimated early on")
    return daily


# --------------------------------------------------------------------------- #
# Degree-day accumulations
# --------------------------------------------------------------------------- #
def afdd(daily: pd.Series) -> pd.Series:
    """Accumulated freezing degree-days: cumsum of max(0, -daily_mean)."""
    return np.maximum(0.0, -daily).cumsum()


def cold_season_anchor(daily: pd.Series, not_before=None) -> pd.Timestamp:
    """Day the season is cumulatively coldest: argmax of cumsum(-daily_mean).

    not_before constrains the anchor to at/after a date (mandatory when the
    caller anchors ATDD after a predicted onset: on a warm winter the
    unconstrained argmax collapses to the first day).
    """
    signed = (-daily).cumsum()
    if not_before is not None:
        signed = signed[signed.index >= pd.Timestamp(not_before)]
        if signed.empty:
            raise ValueError(f"no days at/after not_before={not_before}")
    return signed.idxmax()


def atdd(daily: pd.Series, anchor) -> pd.Series:
    """Accumulated thawing degree-days after `anchor` (exclusive)."""
    spring = daily[daily.index > pd.Timestamp(anchor)]
    return np.maximum(0.0, spring).cumsum()


def value_at_times(daily_cum: pd.Series, times_utc, utc_offset_hours: int) -> np.ndarray:
    """Per-timestamp lookup of a daily cumulative series (step function).

    Each UTC timestamp is mapped to its local-standard day; days before the
    series start read 0.0 (nothing accumulated yet).
    """
    t = pd.to_datetime(pd.Series(list(times_utc)), utc=True)
    days = (t + pd.Timedelta(hours=utc_offset_hours)).dt.floor("D").dt.tz_localize(None)
    looked = daily_cum.reindex(daily_cum.index.union(pd.DatetimeIndex(days.unique()))) \
                      .ffill().reindex(pd.DatetimeIndex(days)).to_numpy()
    return np.nan_to_num(looked, nan=0.0)


def value_at_event(daily_cum: pd.Series, ts_utc, utc_offset_hours: int) -> float:
    """Cumulative value as of the local-standard day of one event timestamp."""
    return float(value_at_times(daily_cum, [ts_utc], utc_offset_hours)[0])


# --------------------------------------------------------------------------- #
# Calibration (TRAIN winters only -- the caller enforces the fold discipline)
# --------------------------------------------------------------------------- #
def _reference_events(results_dir: str | Path, source: str = "usgs_ice_flag") -> pd.DataFrame:
    path = Path(results_dir) / "reference_events.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run fm_ice.evaluation.reference_events first")
    df = pd.read_csv(path)
    return df[df["source"] == source]


def afdd_at_reference_onset(station: str, winter: str, results_dir: str | Path = "results",
                            raw_dir: str | Path = "data/raw") -> float:
    ref = _reference_events(results_dir)
    row = ref[(ref["station"] == station) & (ref["winter"].astype(str) == winter)]
    if row.empty or pd.isna(row.iloc[0]["onset_utc"]):
        raise SystemExit(f"no usgs_ice_flag onset for {station} {winter}")
    daily = load_daily_mean_tmpc(station, winter, raw_dir)
    return value_at_event(afdd(daily), pd.Timestamp(row.iloc[0]["onset_utc"]),
                          station_utc_offset_hours(station))


def calibrate_tau_afdd(calib: list[tuple[str, str]], frac: float = 0.5,
                       results_dir: str | Path = "results",
                       raw_dir: str | Path = "data/raw") -> float:
    """AFDD guard threshold from calibration (train) station-winters only.

    tau = frac * min over calibration winters of AFDD at the reference onset.
    The guard's job is to reject warm-season false positives (AFDD ~ 0 in early
    November), not to predict onset; the min (halved) is the most conservative
    statistic, so a genuine onset is never delayed by the guard.
    """
    vals = [afdd_at_reference_onset(st, w, results_dir, raw_dir) for st, w in calib]
    return float(frac * min(vals))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Self-test on synthetic data, no files needed.
    days = pd.date_range("2024-11-01", "2025-05-15", freq="D")
    n = len(days)
    # warm Nov (+5), freeze Dec-Feb (-10), thaw from Mar (+8)
    temp = np.full(n, 5.0)
    temp[(days >= "2024-12-01") & (days <= "2025-02-28")] = -10.0
    temp[days >= "2025-03-01"] = 8.0
    daily = pd.Series(temp, index=days)

    A = afdd(daily)
    assert A.loc["2024-11-30"] == 0.0                       # nothing frozen yet
    assert A.loc["2024-12-10"] == 100.0                     # 10 days * 10 deg
    assert A.loc["2025-05-15"] == A.loc["2025-02-28"]       # flat after thaw

    anchor = cold_season_anchor(daily)
    assert anchor == pd.Timestamp("2025-02-28"), anchor     # cumulatively coldest day
    T = atdd(daily, anchor)
    assert T.loc["2025-03-10"] == 80.0                      # 10 days * 8 deg

    # warm-winter trap: unconstrained anchor collapses to day 1 if the season
    # never dips; not_before pins it into spring.
    warm = pd.Series(np.full(n, 3.0), index=days)
    assert cold_season_anchor(warm) == days[0]
    assert cold_season_anchor(warm, not_before="2025-03-01") == pd.Timestamp("2025-03-01")

    # per-clip lookup: 06:00 UTC == local midnight CST reads that day's value;
    # pre-season timestamps read 0.
    got = value_at_times(A, ["2024-12-10 06:00:00+00:00", "2024-10-01 06:00:00+00:00"],
                         utc_offset_hours=-6)
    assert got[0] == 100.0 and got[1] == 0.0, got

    # hourly -> daily on the local-standard day: 23:30 local lands on day 1,
    # 00:30 UTC of day 2 (18:30 local day 1) also lands on day 1.
    hourly = pd.DataFrame({
        "datetime_utc": ["2024-11-02 05:30:00+00:00",   # 23:30 local Nov 1
                         "2024-11-02 00:30:00+00:00",   # 18:30 local Nov 1
                         "2024-11-03 06:00:00+00:00"],  # 00:00 local Nov 3
        "tmpc": [-4.0, -2.0, 6.0],
    })
    d = daily_mean_from_hourly(hourly, utc_offset_hours=-6,
                               start=pd.Timestamp("2024-11-01"),
                               end=pd.Timestamp("2024-11-03"))
    assert d.loc["2024-11-01"] == -3.0                      # mean of the two local-day-1 obs
    assert not d.isna().any()                               # Nov 2 gap interpolated

    print("degree_days self-test OK")
