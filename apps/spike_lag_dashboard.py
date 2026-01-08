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
        "columns": ["btc_price", "price_to_beat", "coinbase_price"],
        "time_source": "btc",
        "yaxis_title": "price",
    },
    {
        "name": "BTC delta",
        "columns": ["btc_delta", "coinbase_delta"],
        "time_source": "btc",
        "yaxis_title": "delta",
    },
    {
        "name": "Fair / lag",
        "columns": ["p_fair_up", "dp_fair_up", "lag_react_up", "lag_react_down", "cross_strength"],
        "time_source": "event",
    },
    {
        "name": "Up book (bid/ask)",
        "columns": ["up_bid", "up_ask", "micro_up"],
        "time_source": "up_book",
        "yaxis_title": "price",
        "yaxis_range": [0, 1],
    },
    {
        "name": "Down book (bid/ask)",
        "columns": ["down_bid", "down_ask", "micro_down"],
        "time_source": "down_book",
        "yaxis_title": "price",
        "yaxis_range": [0, 1],
    },
    {
        "name": "Spreads (ticks)",
        "columns": ["spread_up_ticks", "spread_down_ticks"],
        "time_source": "book",
    },
    {
        "name": "Entry gating",
        "columns": ["tau_s", "tau_min_s", "tau_max_s", "tau_flat_s", "btc_tick_age_ms", "max_btc_tick_age_ms"],
        "time_source": "event",
    },
    {
        "name": "Entry thresholds",
        "columns": ["entry_lag_threshold", "taker_lag_threshold", "late_ticks"],
        "time_source": "event",
    },
    {
        "name": "Signals",
        "columns": ["up_signal", "down_signal", "choose_up", "choose_down", "cross_ok", "new_btc_tick"],
        "time_source": "event",
    },
]


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


st.set_page_config(page_title="SpikeLag Dashboard", layout="wide")

st.title("SpikeLag Dashboard (Streamlit + Plotly)")

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
        "SpikeLag JSONL path",
        value="",
        placeholder="/path/to/spike_lag.jsonl",
    )
    compare_path = st.text_input(
        "Compare JSONL path (optional)",
        value="",
        placeholder="/path/to/spike_lag.jsonl",
    )
    strategy = st.text_input("Strategy", value="spike_lag")
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

if not input_path:
    st.info("Set spike_lag JSONL path in the sidebar.")
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

base_time = _base_time(df)

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
                compare_base = _base_time(compare_df)
            except Exception as exc:
                st.error(f"Failed to load compare data: {exc}")
    else:
        st.warning(f"Compare file not found: {compare_file}")

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

st.subheader("Standard plots")
selected = st.multiselect(
    "Select plots",
    [tpl["name"] for tpl in PLOT_TEMPLATES],
    default=[
        "BTC price vs strike",
        "Up book (bid/ask)",
        "Down book (bid/ask)",
        "Fair / lag",
    ],
)

for tpl in PLOT_TEMPLATES:
    if tpl["name"] not in selected:
        continue
    width = int(base_width * xscale)
    fig = _plot_lines(
        df,
        tpl["columns"],
        tpl["name"],
        tpl["time_source"],
        yaxis_title=tpl.get("yaxis_title"),
        yaxis_range=tpl.get("yaxis_range"),
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
            tpl["time_source"],
            yaxis_title=tpl.get("yaxis_title"),
            yaxis_range=tpl.get("yaxis_range"),
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
        options=["event", "btc", "book", "up_book", "down_book", "log"],
        index=0,
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
