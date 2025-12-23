# api/fred_api.py
"""
Vercel Serverless API: FRED monthly panel (past N years, default 20)

Endpoints:
- GET /api/fred_api/health
- GET /api/fred_api/fred/panel?years=20&api_key=...&timeout=30&use_cache=true

Notes for Vercel:
- Do NOT run uvicorn here; Vercel will host the ASGI app.
- FastAPI app object must be named `app`.
- In Vercel, this file lives under /api so it becomes a serverless function.

Env:
- FRED_API_KEY: optional default key if query param api_key is omitted
- CACHE_TTL_SEC: optional cache TTL seconds (default 3600)

Behavior:
- Builds a full monthly calendar from (current month - years) to current month (inclusive)
- For each non-quarterly series: request monthly frequency with aggregation_method=avg, then ffill
- For GDPC1 (quarterly): map to quarter-end month and backfill within quarter, then ffill across time
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse


# -----------------------------
# Config
# -----------------------------
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES_IDS: List[str] = [
    "RSXFS", "PCEPI", "CPIAUCSL", "PPIACO", "JTSJOL", "PAYEMS", "UNRATE",
    "ICSA", "CCSA", "DFEDTARU", "DFEDTARL", "FEDFUNDS",
    "WLTLECL", "WALCL", "DGS2", "DGS10", "DTWEXBGS", "GDPC1"
]
QUARTERLY_SERIES = {"GDPC1"}

CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "3600"))

# Simple in-memory cache: (years, api_key) -> (expires_at_epoch, payload_dict)
_CACHE: Dict[Tuple[int, str], Tuple[float, Dict]] = {}


# -----------------------------
# Data logic
# -----------------------------
@dataclass(frozen=True)
class TimeRange:
    start_ms: pd.Timestamp  # month-start
    end_ms: pd.Timestamp    # month-start (current month)


def compute_timerange(years: int) -> TimeRange:
    today = date.today()
    end = pd.Timestamp(today.year, today.month, 1)
    start = end - relativedelta(years=years)
    start = pd.Timestamp(start.year, start.month, 1)
    return TimeRange(start_ms=start, end_ms=end)


def build_monthly_calendar(tr: TimeRange) -> pd.DatetimeIndex:
    return pd.date_range(tr.start_ms, tr.end_ms, freq="MS")


def parse_value(x) -> Optional[float]:
    """FRED uses '.' for missing."""
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == ".":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fred_fetch_observations(
    series_id: str,
    observation_start: str,
    api_key: str,
    *,
    frequency: Optional[str],
    aggregation_method: Optional[str],
    session: requests.Session,
    timeout: int,
    max_retries: int = 3,
) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
    }
    if frequency is not None:
        params["frequency"] = frequency
    if aggregation_method is not None:
        params["aggregation_method"] = aggregation_method

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(BASE_URL, params=params, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            obs = payload.get("observations", [])
            # Keep only required columns; missing series returns empty obs array
            df = pd.DataFrame(obs)
            if df.empty:
                return pd.DataFrame({"date": [], "value": []})
            return df[["date", "value"]].copy()
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(0.7 * attempt)
            else:
                raise RuntimeError(f"FRED fetch failed for {series_id}: {last_err}") from last_err


def to_monthly_series_ffill(obs_df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """Left-align to monthly calendar and ffill to handle publication lags."""
    if obs_df is None or obs_df.empty:
        return pd.Series(index=calendar, dtype="float64")

    df = obs_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = df["value"].map(parse_value)

    s = df.dropna(subset=["date"]).set_index("date")["value"].sort_index()
    # Normalize to month-start
    s.index = s.index.to_period("M").to_timestamp("MS")
    # Duplicates within the same month -> keep last
    s = s[~s.index.duplicated(keep="last")]

    return s.reindex(calendar).ffill()


def gdpc1_quarterly_to_monthly(obs_df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """
    GDPC1 quarterly handling:
    - Map each quarter to quarter-end month (Mar/Jun/Sep/Dec) at month-start index
    - Value appears on quarter-end month; then backfill within that quarter to all months in the quarter
    - Finally ffill across time for release lags
    """
    s = pd.Series(index=calendar, dtype="float64")
    if obs_df is None or obs_df.empty:
        return s

    df = obs_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = df["value"].map(parse_value)
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return s

    # Convert observation date -> quarter -> quarter-end month-start
    q_period = df["date"].dt.to_period("Q")
    q_end_ms = q_period.asfreq("M", how="end").dt.to_timestamp("MS")

    q_end_series = pd.Series(df["value"].values, index=q_end_ms).sort_index()
    q_end_series = q_end_series[~q_end_series.index.duplicated(keep="last")]

    idx = q_end_series.index.intersection(calendar)
    s.loc[idx] = q_end_series.loc[idx].astype("float64")

    # Backfill within each quarter only (no cross-quarter leakage)
    s = s.groupby(calendar.to_period("Q")).apply(lambda x: x.bfill())
    s.index = calendar

    # Carry forward across time (publication lag)
    return s.ffill()


def build_panel_json(panel: pd.DataFrame, tr: TimeRange, indicators: List[str]) -> Dict:
    time_range_str = f"{tr.start_ms.strftime('%Y-%m')} 至 {tr.end_ms.strftime('%Y-%m')}"
    rows = []
    for dt, row in panel.iterrows():
        values = {}
        for k in indicators:
            v = row.get(k)
            values[k] = None if pd.isna(v) else float(v)
        rows.append({"date": dt.strftime("%Y-%m-%d"), "values": values})

    return {
        "metadata": {
            "time_range": time_range_str,
            "frequency": "monthly",
            "indicators": indicators,
        },
        "data": rows,
    }


def build_fred_json(api_key: str, years: int, timeout: int) -> Dict:
    tr = compute_timerange(years)
    calendar = build_monthly_calendar(tr)
    observation_start = tr.start_ms.strftime("%Y-%m-%d")

    sess = requests.Session()
    series_map: Dict[str, pd.Series] = {}

    for sid in SERIES_IDS:
        if sid in QUARTERLY_SERIES:
            obs = fred_fetch_observations(
                sid,
                observation_start=observation_start,
                api_key=api_key,
                frequency=None,
                aggregation_method=None,
                session=sess,
                timeout=timeout,
            )
            series_map[sid] = gdpc1_quarterly_to_monthly(obs, calendar)
        else:
            obs = fred_fetch_observations(
                sid,
                observation_start=observation_start,
                api_key=api_key,
                frequency="m",
                aggregation_method="avg",
                session=sess,
                timeout=timeout,
            )
            series_map[sid] = to_monthly_series_ffill(obs, calendar)

    panel = pd.DataFrame(series_map, index=calendar)
    return build_panel_json(panel, tr, SERIES_IDS)


def _cache_get(years: int, api_key: str) -> Optional[Dict]:
    key = (years, api_key)
    hit = _CACHE.get(key)
    if not hit:
        return None
    expires_at, payload = hit
    if time.time() >= expires_at:
        _CACHE.pop(key, None)
        return None
    return payload


def _cache_set(years: int, api_key: str, payload: Dict) -> None:
    key = (years, api_key)
    _CACHE[key] = (time.time() + CACHE_TTL_SEC, payload)


# -----------------------------
# FastAPI app (Vercel entry)
# -----------------------------
app = FastAPI(title="FRED Monthly Panel API", version="1.0.0")


@app.get("/api/fred_api/health")
def health():
    return {"status": "ok"}


@app.get("/api/fred_api/fred/panel")
def fred_panel(
    years: int = Query(20, ge=1, le=60, description="Lookback window in years"),
    api_key: Optional[str] = Query(None, description="FRED API key; if omitted uses env FRED_API_KEY"),
    timeout: int = Query(30, ge=5, le=120, description="Timeout (seconds) per request to FRED"),
    use_cache: bool = Query(True, description="Use in-memory cache"),
):
    key = api_key or os.getenv("FRED_API_KEY")
    if not key or key == "YOUR_FRED_API_KEY":
        raise HTTPException(
            status_code=400,
            detail="Missing FRED API key. Pass ?api_key=... or set env FRED_API_KEY in Vercel.",
        )

    if use_cache:
        cached = _cache_get(years, key)
        if cached is not None:
            return JSONResponse(content=cached)

    try:
        payload = build_fred_json(api_key=key, years=years, timeout=timeout)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to build panel from FRED: {e}")

    if use_cache:
        _cache_set(years, key, payload)

    return JSONResponse(content=payload)


# IMPORTANT:
# Do NOT add uvicorn.run(...) here. Vercel will start the ASGI app.
