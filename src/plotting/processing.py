from __future__ import annotations

import pandas as pd
import numpy as np
import sys


MIN_VALID_MS = 0  # Unix epoch and later
MAX_VALID_MS = 9_223_372_036_854  # pandas max Timestamp in ms (2262-04-11)


def _safe_ms_to_datetime(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.where(series.between(MIN_VALID_MS, MAX_VALID_MS))
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return pd.to_datetime(
                series.astype("float64"), unit="ms", utc=True, errors="coerce"
            )
    except FloatingPointError:
        return pd.to_datetime(
            pd.Series([np.nan] * len(series), index=series.index),
            unit="ms",
            utc=True,
            errors="coerce",
        )


def normalize_records(records: list[dict]) -> pd.DataFrame:
    df = pd.json_normalize(records, sep=".")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        candidates = []
        if "ts_ms" in df.columns:
            candidates.append("ts_ms")
        if "event_ts_ms.log" in df.columns:
            candidates.append("event_ts_ms.log")
        if "engine_ts_ms" in df.columns:
            candidates.append("engine_ts_ms")
        if "event_ts_ms.book" in df.columns:
            candidates.append("event_ts_ms.book")

        chosen = None
        for col in candidates:
            series = _safe_ms_to_datetime(df[col])
            if series.notna().any():
                df["timestamp"] = series
                chosen = col
                break

        if chosen is None:
            raise ValueError("Missing usable timestamp (ts_ms/event_ts_ms.log/engine_ts_ms)")
    if "__line__" in df.columns:
        missing_ts = df["timestamp"].isna()
        if missing_ts.any():
            bad_lines = (
                df.loc[missing_ts, "__line__"]
                .dropna()
                .astype(int)
                .tolist()
            )
            if bad_lines:
                sample = ", ".join(str(x) for x in bad_lines[:20])
                suffix = "..." if len(bad_lines) > 20 else ""
                print(
                    f"[warn] Invalid timestamps at lines: {sample}{suffix}",
                    file=sys.stderr,
                )

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        raise ValueError("All timestamps are invalid or missing")

    return df.reset_index(drop=True)


def maybe_downsample(
    df: pd.DataFrame,
    *,
    max_points: int = 100_000,
    rule: str | None = "50ms",
) -> pd.DataFrame:
    if rule is None or len(df) <= max_points:
        return df
    if "timestamp" not in df.columns:
        return df

    df_sorted = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df_sorted.empty:
        return df

    grouped = df_sorted.groupby(pd.Grouper(key="timestamp", freq=rule))
    resampled = grouped.tail(1)
    resampled = resampled.dropna(how="all")
    if resampled.empty:
        return df
    return resampled.reset_index(drop=True)
