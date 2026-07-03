"""Config loading and small date helpers used across the data scripts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    """Load a yaml file from configs/ by name, e.g. load_yaml('stations.yaml')."""
    path = CONFIG_DIR / name if not Path(name).is_absolute() else Path(name)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def station_utc_offset_hours(station: str) -> int:
    """usgs_dv_utc_offset_hours for a station: per-station override wins over
    the default. USGS daily values (and our reference events at local midnight)
    use the LOCAL STANDARD-TIME day, which differs per timezone -- e.g.
    mohawk_schenectady is EST (-5) while the WI/ND stations are CST (-6)."""
    cfg = load_yaml("stations.yaml")
    st = cfg["stations"][station]
    return int(st.get("usgs_dv_utc_offset_hours",
                      cfg["defaults"]["usgs_dv_utc_offset_hours"]))


def winter_bounds(winter: str, start_md: str, end_md: str) -> tuple[date, date]:
    """Map a winter label like '2024-2025' to concrete start/end dates.

    start_md belongs to the first year, end_md to the second year.
    """
    y0, y1 = (int(p) for p in winter.split("-"))
    sm, sd = (int(p) for p in start_md.split("-"))
    em, ed = (int(p) for p in end_md.split("-"))
    return date(y0, sm, sd), date(y1, em, ed)
