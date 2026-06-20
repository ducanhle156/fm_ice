"""Small HTTP helpers shared by the downloaders.

Dependency-light on purpose: only `requests`. Retries with backoff, a polite
default User-Agent, and an optional USGS API key read from the environment.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests

DEFAULT_UA = "fm_ice-data-downloader/0.1 (NDSU ICE Lab; research use)"


def session(api_key_env: str = "USGS_API_KEY") -> requests.Session:
    """Return a configured Session. If the env var is set, send it as X-Api-Key.

    The HIVIS /cameras and /listFiles endpoints document a 403 when a key is
    missing, but currently serve low-volume unauthenticated requests. Get a free
    key at https://api.waterdata.usgs.gov/signup/ if you hit OVER_RATE_LIMIT.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": DEFAULT_UA, "Accept": "application/json"})
    key = os.environ.get(api_key_env)
    if key:
        s.headers.update({"X-Api-Key": key})
    return s


def get(
    s: requests.Session,
    url: str,
    params: Optional[dict] = None,
    *,
    retries: int = 5,
    backoff: float = 2.0,
    timeout: int = 60,
    stream: bool = False,
) -> requests.Response:
    """GET with exponential backoff on 429/5xx and transient network errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, timeout=timeout, stream=stream)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r
        except (requests.RequestException,) as e:
            last_exc = e
            sleep = backoff ** attempt
            time.sleep(min(sleep, 30))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}\n  last error: {last_exc}")
