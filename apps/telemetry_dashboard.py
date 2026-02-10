from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting.io import filter_records, load_jsonl
from plotting.processing import maybe_downsample, normalize_records
from plotting.plots import _base_time, _elapsed_seconds, _select_time_series

MIN_VALID_MS = 0
MAX_VALID_MS = 9_223_372_036_854

PLOT_TEMPLATES = [
    {
        "name": "BTC price vs strike",
        "columns": ["btc.price", "btc.price_to_beat"],
        "time_source": "btc",
        "yaxis_title": "price",
    },
    {
        "name": "BTC delta",
        "columns": ["btc.delta", "btc.delta_pct"],
        "time_source": "btc",
        "yaxis_title": "delta",
    },
    {
        "name": "Fair probabilities",
        "columns": ["btc.p_fair_up", "btc.p_fair_down"],
        "time_source": "btc",
        "yaxis_title": "probability",
        "yaxis_range": [0, 1],
    },
    {
        "name": "Up L1 bid/ask",
        "columns": ["up_l1.bid_p", "up_l1.ask_p", "up_l1.mid"],
        "time_source": "up_book",
        "yaxis_title": "price",
        "yaxis_range": [0, 1],
    },
    {
        "name": "Down L1 bid/ask",
        "columns": ["down_l1.bid_p", "down_l1.ask_p", "down_l1.mid"],
        "time_source": "down_book",
        "yaxis_title": "price",
        "yaxis_range": [0, 1],
    },
    {
        "name": "Shock / impulse",
        "columns": ["shock.shock_ratio", "shock.move_5s", "btc.z_impulse_r1", "btc.z_delta"],
        "time_source": "event",
    },
    {
        "name": "Fair gap",
        "columns": ["btc.fair_gap_up", "btc.fair_gap_down"],
        "time_source": "event",
        "yaxis_title": "gap",
    },
    {
        "name": "Skew up/down",
        "columns": ["skew_up_down"],
        "time_source": "book",
    },
]


def _price_source_map(df: pd.DataFrame) -> dict[str, str]:
    sources: dict[str, str] = {}
    for col in df.columns:
        if col.startswith("prices.by_source.") and col.endswith(".price"):
            name = col[len("prices.by_source.") : -len(".price")]
            sources[name] = col
        if col.startswith("telemetry_state.extra_prices.") and col.endswith(".last_sample.price"):
            name = col[len("telemetry_state.extra_prices.") : -len(".last_sample.price")]
            sources[name] = col
    return sources


def _window_500ms_columns(df: pd.DataFrame) -> list[str]:
    sources: set[str] = set()
    prefix = "prices.windows."
    needle = ".window_500ms."
    for col in df.columns:
        if col.startswith(prefix) and needle in col:
            rest = col[len(prefix) :]
            source = rest.split(needle, 1)[0]
            if source:
                sources.add(source)
    columns: list[str] = []
    for source in sorted(sources):
        for metric in ("delta_from_max", "delta_from_min"):
            col = f"prices.windows.{source}.window_500ms.{metric}"
            if col in df.columns:
                columns.append(col)
    return columns


def _anchored_source_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    estimates: list[str] = []
    deltas: list[str] = []
    prefix = "prices.anchored.sources."
    for col in df.columns:
        if col.startswith(prefix) and col.endswith(".last_estimate"):
            estimates.append(col)
        if col.startswith(prefix) and col.endswith(".last_delta"):
            deltas.append(col)
    return sorted(estimates), sorted(deltas)


def _load_dataframe(path: str, strategy: str, market: str | None, rule: str | None, max_points: int) -> pd.DataFrame:
    records = load_jsonl(path)
    filtered = filter_records(records, strategy=strategy, market_slug=market)
    df = normalize_records(filtered)
    df = maybe_downsample(df, rule=rule, max_points=max_points)
    return df


@st.cache_data(show_spinner=False)
def _cached_dataframe(path: str, strategy: str, market: str | None, rule: str | None, max_points: int) -> pd.DataFrame:
    return _load_dataframe(path, strategy, market, rule, max_points)


@st.cache_data(show_spinner=False)
def _cached_raw_df(path: str, strategy: str, market: str | None) -> pd.DataFrame:
    records = load_jsonl(path)
    filtered = filter_records(records, strategy=strategy, market_slug=market)
    return pd.json_normalize(filtered, sep=".")


def _find_invalid_ts_lines(df: pd.DataFrame) -> dict[str, list[int]]:
    if "__line__" not in df.columns:
        return {}

    ts_columns = [
        col
        for col in df.columns
        if col == "ts_ms"
        or col.startswith("event_ts_ms")
        or col.endswith(".ts_ms")
    ]
    invalid: dict[str, list[int]] = {}
    for col in ts_columns:
        raw = df[col]
        series = pd.to_numeric(raw, errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan)
        missing = raw.isna()
        non_numeric = series.isna() & ~missing
        out_of_range = series.notna() & ~series.between(MIN_VALID_MS, MAX_VALID_MS)
        mask = non_numeric | out_of_range
        if mask.any():
            lines = (
                df.loc[mask, "__line__"]
                .dropna()
                .astype(int)
                .tolist()
            )
            if lines:
                invalid[col] = lines
    return invalid


def _engine_base_time(df: pd.DataFrame) -> pd.Timestamp | None:
    if "engine_ts_ms" not in df.columns:
        return None
    series = pd.to_numeric(df["engine_ts_ms"], errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    if not series.notna().any():
        return None
    return pd.to_datetime(series.min(), unit="ms", utc=True, errors="coerce")


def _plot_lines(
    df: pd.DataFrame,
    columns: Iterable[str],
    title: str,
    time_source: str,
    *,
    yaxis_title: str | None = None,
    yaxis_range: list[float] | None = None,
    baseline_col: str | None = None,
    base_time: pd.Timestamp | None = None,
    line_dash: str | None = None,
    name_prefix: str | None = None,
    fig: go.Figure | None = None,
    x_tick: int = 5,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    ts = _select_time_series(df, time_source)
    x = _elapsed_seconds(ts, base_time)

    baseline = None
    if baseline_col and baseline_col in df.columns:
        baseline = pd.to_numeric(df[baseline_col], errors="coerce")

    customdata = None
    if "__line__" in df.columns:
        customdata = df["__line__"].astype("Int64").tolist()

    for col in columns:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        if baseline is not None:
            y = y - baseline
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{name_prefix}{col}" if name_prefix else col,
                line={"dash": line_dash} if line_dash else None,
                customdata=customdata,
                hovertemplate=(
                    f"line=%{{customdata}}<br>t=%{{x}}s<br>{col}=%{{y}}<extra></extra>"
                    if customdata is not None
                    else None
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="seconds",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    if yaxis_range is not None:
        fig.update_yaxes(range=yaxis_range)
    return fig


st.set_page_config(page_title="Telemetry Dashboard", layout="wide")

st.title("Telemetry Dashboard (Streamlit + Plotly)")

st.markdown(
    """
<style>
div[data-testid="stPlotlyChart"] {
  overflow-x: auto;
}
div[data-testid="stPlotlyChart"] > div {
  width: max-content;
}
</style>
""",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.header("Data")
    input_path = st.text_input(
        "Telemetry JSONL path",
        value="",
        placeholder="/path/to/telemetry.jsonl",
    )
    compare_path = st.text_input(
        "Compare JSONL path (optional)",
        value="",
        placeholder="/path/to/telemetry.jsonl",
    )
    strategy = st.text_input("Strategy", value="telemetry")
    market = st.text_input("Market slug (optional)", value="")
    downsample = st.text_input("Downsample rule", value="50ms")
    if downsample.strip().lower() == "none":
        downsample = None
    max_points = st.number_input("Max points", value=500000, step=50000)
    st.header("Display")
    x_tick = st.number_input("X tick (seconds)", value=5, step=1)
    height = st.number_input("Chart height", value=500, step=50)
    base_width = st.number_input("Base chart width (px)", value=1200, step=100)
    xscale = st.number_input("X scale", value=1.0, step=0.25)
    time_axis = st.selectbox("Time axis", options=["engine_ts_ms", "auto"], index=0)

if not input_path:
    st.info("Set telemetry JSONL path in the sidebar.")
    st.stop()

path = Path(input_path)
if not path.exists():
    st.error(f"File not found: {path}")
    st.stop()

with st.spinner("Loading data..."):
    try:
        df = _cached_dataframe(
            str(path),
            strategy=strategy,
            market=market or None,
            rule=downsample,
            max_points=int(max_points),
        )
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

if time_axis == "engine_ts_ms":
    base_time = _engine_base_time(df) or _base_time(df)
else:
    base_time = _base_time(df)

dynamic_templates = []
source_map = _price_source_map(df)
price_cols = sorted(source_map.values())
window_500_cols = _window_500ms_columns(df)
anchored_est_cols, anchored_delta_cols = _anchored_source_columns(df)
primary_source = None
if "prices.primary_source" in df.columns:
    primary_series = df["prices.primary_source"].dropna()
    if not primary_series.empty:
        primary_source = str(primary_series.iloc[0])

if price_cols:
    dynamic_templates.append(
        {
            "name": "Price sources (all)",
            "columns": price_cols,
            "time_source": "engine",
            "yaxis_title": "BTC price",
        }
    )
    dynamic_templates.append(
        {
            "name": "Price sources (raw)",
            "columns": price_cols + ["btc.price", "btc.price_to_beat"],
            "time_source": "btc",
            "yaxis_title": "BTC price",
        }
    )
    dynamic_templates.append(
        {
            "name": "Price sources delta vs K",
            "columns": price_cols,
            "time_source": "btc",
            "yaxis_title": "price - K",
            "baseline_col": "btc.price_to_beat",
        }
    )

if window_500_cols:
    dynamic_templates.append(
        {
            "name": "Price windows 500ms (delta)",
            "columns": window_500_cols,
            "time_source": "engine",
            "yaxis_title": "delta from max/min",
            "force_engine": True,
        }
    )

anchored_base_cols = []
if "prices.anchored.anchor_price" in df.columns:
    anchored_base_cols.append("prices.anchored.anchor_price")
if "prices.anchored.median_estimate" in df.columns:
    anchored_base_cols.append("prices.anchored.median_estimate")

if anchored_base_cols or anchored_est_cols:
    dynamic_templates.append(
        {
            "name": "Anchored estimates (RTDS vs sources)",
            "columns": anchored_base_cols + anchored_est_cols,
            "time_source": "engine",
            "yaxis_title": "BTC price",
            "force_engine": True,
        }
    )

if anchored_base_cols and "btc.price_to_beat" in df.columns:
    dynamic_templates.append(
        {
            "name": "Anchored delta vs strike",
            "columns": anchored_base_cols,
            "time_source": "engine",
            "yaxis_title": "price - K",
            "baseline_col": "btc.price_to_beat",
            "force_engine": True,
        }
    )

if anchored_delta_cols:
    dynamic_templates.append(
        {
            "name": "Anchored per-source delta",
            "columns": anchored_delta_cols,
            "time_source": "engine",
            "yaxis_title": "delta",
            "force_engine": True,
        }
    )

    rtds_key = None
    if primary_source and primary_source in source_map:
        rtds_key = primary_source
    if rtds_key is None:
        for key in source_map:
            if "rtds" in key:
                rtds_key = key
                break

    if rtds_key is not None:
        rtds_col = source_map[rtds_key]
        targets = {
            "coinbase": None,
            "pyth": None,
            "tiingo": None,
        }
        for key, col in source_map.items():
            for target in list(targets.keys()):
                if target in key and targets[target] is None:
                    targets[target] = col

        for target, col in targets.items():
            if col is None:
                continue
            label = target.upper()
            dynamic_templates.append(
                {
                    "name": f"RTDS vs {label} (price)",
                    "columns": [rtds_col, col],
                    "time_source": "btc",
                    "yaxis_title": "BTC price",
                }
            )
            dynamic_templates.append(
                {
                    "name": f"RTDS vs {label} (delta)",
                    "columns": [col],
                    "time_source": "btc",
                    "yaxis_title": "price - RTDS",
                    "baseline_col": rtds_col,
                }
            )

with st.expander("Invalid timestamps (scan raw JSONL)", expanded=False):
    if st.button("Scan for invalid ts_ms"):
        raw_df = _cached_raw_df(str(path), strategy=strategy, market=market or None)
        invalid = _find_invalid_ts_lines(raw_df)
        if not invalid:
            st.success("No invalid timestamps found.")
        else:
            for col, lines in invalid.items():
                sample = ", ".join(str(x) for x in lines[:30])
                suffix = "..." if len(lines) > 30 else ""
                st.write(f"{col}: {len(lines)} lines (sample: {sample}{suffix})")

compare_df = None
compare_base = None
if compare_path:
    compare_file = Path(compare_path)
    if compare_file.exists():
        with st.spinner("Loading compare data..."):
            try:
                compare_df = _cached_dataframe(
                    str(compare_file),
                    strategy=strategy,
                    market=market or None,
                    rule=downsample,
                    max_points=int(max_points),
                )
                if time_axis == "engine_ts_ms":
                    compare_base = _engine_base_time(compare_df) or _base_time(compare_df)
                else:
                    compare_base = _base_time(compare_df)
            except Exception as exc:
                st.error(f"Failed to load compare data: {exc}")
    else:
        st.warning(f"Compare file not found: {compare_file}")

st.subheader("Standard plots")
with st.expander("Detected price source columns", expanded=False):
    if source_map:
        st.write(source_map)
    else:
        st.info("No prices.by_source.*.price or telemetry_state.extra_prices.*.last_sample.price columns found.")

default_plots = ["BTC price vs strike", "Up L1 bid/ask", "Down L1 bid/ask"]
if price_cols:
    default_plots.extend(
        ["Price sources (all)", "Price sources (raw)", "Price sources delta vs K"]
    )
    for tpl in dynamic_templates:
        if tpl["name"].startswith("RTDS vs"):
            default_plots.append(tpl["name"])
if window_500_cols:
    default_plots.append("Price windows 500ms (delta)")
if anchored_base_cols or anchored_est_cols:
    default_plots.append("Anchored estimates (RTDS vs sources)")
if anchored_base_cols and "btc.price_to_beat" in df.columns:
    default_plots.append("Anchored delta vs strike")
if anchored_delta_cols:
    default_plots.append("Anchored per-source delta")

selected = st.multiselect(
    "Select plots",
    [tpl["name"] for tpl in (PLOT_TEMPLATES + dynamic_templates)],
    default=default_plots,
)

for tpl in (PLOT_TEMPLATES + dynamic_templates):
    if tpl["name"] not in selected:
        continue
    width = int(base_width * xscale)
    if tpl.get("force_engine"):
        time_source = "engine"
    else:
        time_source = "engine" if time_axis == "engine_ts_ms" else tpl["time_source"]
    fig = _plot_lines(
        df,
        tpl["columns"],
        tpl["name"],
        time_source,
        yaxis_title=tpl.get("yaxis_title"),
        yaxis_range=tpl.get("yaxis_range"),
        baseline_col=tpl.get("baseline_col"),
        base_time=base_time,
        x_tick=int(x_tick),
        height=int(height),
        width=width,
    )
    if compare_df is not None:
        fig = _plot_lines(
            compare_df,
            tpl["columns"],
            tpl["name"],
            time_source,
            yaxis_title=tpl.get("yaxis_title"),
            yaxis_range=tpl.get("yaxis_range"),
            baseline_col=tpl.get("baseline_col"),
            base_time=compare_base,
            line_dash="dash",
            name_prefix="B:",
            fig=fig,
            x_tick=int(x_tick),
            height=int(height),
            width=width,
        )
    st.plotly_chart(fig, width=width, config={"responsive": False})

st.subheader("Custom plot")
with st.expander("Custom columns"):
    columns = st.multiselect("Columns", options=sorted(df.columns))
    time_source = st.selectbox(
        "Time source",
        options=["engine", "event", "btc", "book", "up_book", "down_book", "log"],
        index=0 if time_axis == "engine_ts_ms" else 1,
    )
    y_min = st.text_input("Y min (optional)", value="")
    y_max = st.text_input("Y max (optional)", value="")

if columns:
    y_range = None
    if y_min.strip() and y_max.strip():
        try:
            y_range = [float(y_min), float(y_max)]
        except ValueError:
            st.warning("Invalid Y range, ignoring.")
    width = int(base_width * xscale)
    fig = _plot_lines(
        df,
        columns,
        "Custom plot",
        time_source,
        base_time=base_time,
        yaxis_range=y_range,
        x_tick=int(x_tick),
        height=int(height),
        width=width,
    )
    if compare_df is not None:
        fig = _plot_lines(
            compare_df,
            columns,
            "Custom plot",
            time_source,
            base_time=compare_base,
            line_dash="dash",
            name_prefix="B:",
            fig=fig,
            x_tick=int(x_tick),
            height=int(height),
            width=width,
        )
    st.plotly_chart(fig, width=width, config={"responsive": False})
