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
    """GET with exponential backoff on 429/5xx and transient network errors.

    Non-retryable client errors (4xx other than 429, e.g. a 404 for a file the
    listing advertised but S3 does not actually have) are raised immediately as
    requests.HTTPError so the caller can skip that item without burning retries.
    """
    RETRYABLE = (429, 500, 502, 503, 504)
    last_exc = None
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, timeout=timeout, stream=stream)
            if r.status_code in RETRYABLE:
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            # Only retry the retryable statuses; surface real 4xx right away.
            status = e.response.status_code if e.response is not None else None
            if status not in RETRYABLE:
                raise
            last_exc = e
            time.sleep(min(backoff ** attempt, 30))
        except requests.RequestException as e:
            last_exc = e
            time.sleep(min(backoff ** attempt, 30))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}\n  last error: {last_exc}")
