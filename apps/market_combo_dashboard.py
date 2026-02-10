from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting.io import load_jsonl
from plotting.processing import maybe_downsample, _safe_ms_to_datetime


def _book_timestamp(df: pd.DataFrame) -> pd.Series:
    if "event_ts_ms.book" in df.columns:
        return df["event_ts_ms.book"]
    if "event_ts_ms.book_up" in df.columns or "event_ts_ms.book_down" in df.columns:
        up = df.get("event_ts_ms.book_up")
        down = df.get("event_ts_ms.book_down")
        if up is not None and down is not None:
            return pd.concat([up, down], axis=1).max(axis=1, skipna=True)
        return up if up is not None else down
    return df.get("ts_ms")


def _telemetry_book_timestamp(df: pd.DataFrame, side: str) -> pd.Series:
    side_key = side.lower()
    if side_key == "book":
        if "event_ts_ms.book" in df.columns:
            return df["event_ts_ms.book"]
        up = df.get("event_ts_ms.book_up")
        down = df.get("event_ts_ms.book_down")
        if up is not None and down is not None:
            return pd.concat([up, down], axis=1).max(axis=1, skipna=True)
        if up is not None:
            return up
        if down is not None:
            return down
    if side_key == "down":
        if "event_ts_ms.book_down" in df.columns:
            return df["event_ts_ms.book_down"]
        if "down_l1.ts_ms" in df.columns:
            return df["down_l1.ts_ms"]
    else:
        if "event_ts_ms.book_up" in df.columns:
            return df["event_ts_ms.book_up"]
        if "up_l1.ts_ms" in df.columns:
            return df["up_l1.ts_ms"]
    if "event_ts_ms.book" in df.columns:
        return df["event_ts_ms.book"]
    return df.get("ts_ms")


def _telemetry_price_sources(df: pd.DataFrame) -> tuple[list[str], str | None]:
    source_map: dict[str, str] = {}
    cols: list[str] = []
    for col in df.columns:
        if col.startswith("prices.by_source.") and col.endswith(".price"):
            name = col[len("prices.by_source.") : -len(".price")]
            source_map[name] = col
            cols.append(col)
        if col.startswith("telemetry_state.extra_prices.") and col.endswith(".last_sample.price"):
            name = col[len("telemetry_state.extra_prices.") : -len(".last_sample.price")]
            source_map[name] = col
            cols.append(col)
        if col.startswith("prices.extra_avg.") and col.endswith(".price"):
            cols.append(col)

    cols = sorted(set(cols))

    primary_col = None
    if "prices.primary_source" in df.columns:
        primary_series = df["prices.primary_source"].dropna()
        if not primary_series.empty:
            key = str(primary_series.iloc[0])
            primary_col = source_map.get(key)
    if primary_col is None:
        for key, col in source_map.items():
            if "rtds" in key:
                primary_col = col
                break
    if primary_col is None and cols:
        primary_col = cols[0]

    return cols, primary_col


def _btc_timestamp(df: pd.DataFrame) -> pd.Series:
    if "event_ts_ms.btc" in df.columns:
        return df["event_ts_ms.btc"]
    if "btc_ts_ms" in df.columns:
        return df["btc_ts_ms"]
    return df.get("ts_ms")


def _engine_timestamp(df: pd.DataFrame) -> pd.Series:
    if "engine_ts_ms" in df.columns:
        return df["engine_ts_ms"]
    return df.get("ts_ms")


def _strategy_engine_series(df: pd.DataFrame) -> pd.Series:
    if "engine_ts_ms" in df.columns:
        series = pd.to_numeric(df["engine_ts_ms"], errors="coerce")
        if series.notna().any():
            return series
    if "decision_ts_ms" in df.columns:
        return pd.to_numeric(df["decision_ts_ms"], errors="coerce")
    return pd.Series(dtype="float64")


def _elapsed_seconds(ts: pd.Series, base_time: pd.Timestamp) -> pd.Series:
    dt = _safe_ms_to_datetime(pd.to_numeric(ts, errors="coerce"))
    return (dt - base_time).dt.total_seconds()


def _engine_base_time(
    df_strategy: pd.DataFrame,
    df_telemetry: pd.DataFrame | None,
    exec_records: list[dict],
) -> float | None:
    candidates = []

    if "engine_ts_ms" in df_strategy.columns:
        strat_vals = pd.to_numeric(df_strategy["engine_ts_ms"], errors="coerce")
        if strat_vals.notna().any():
            candidates.append(strat_vals.min())

    if df_telemetry is not None and "engine_ts_ms" in df_telemetry.columns:
        tele_vals = pd.to_numeric(df_telemetry["engine_ts_ms"], errors="coerce")
        if tele_vals.notna().any():
            candidates.append(tele_vals.min())

    exec_engine = [
        rec.get("engine_ts_ms") for rec in exec_records if rec.get("engine_ts_ms") is not None
    ]
    if exec_engine:
        exec_vals = pd.to_numeric(pd.Series(exec_engine), errors="coerce")
        if exec_vals.notna().any():
            candidates.append(exec_vals.min())

    if candidates:
        return min(candidates)

    return None


def _infer_engine_scale_seconds(
    raw_series: pd.Series,
    reference_span_ms: float | None,
) -> float:
    raw = pd.to_numeric(raw_series, errors="coerce").dropna()
    if raw.empty:
        return 0.001
    span = raw.max() - raw.min()
    if reference_span_ms and reference_span_ms > 0 and span > 0:
        ratio = span / reference_span_ms
        if 0.5 <= ratio <= 2.0:
            return 1.0 / 1000.0  # ms -> s
        if 500 <= ratio <= 2000:
            return 1.0 / 1_000_000.0  # us -> s
        if 0.0005 <= ratio <= 0.002:
            return 1.0  # already seconds
    max_val = raw.max()
    if max_val > 1e14:
        return 1.0 / 1_000_000.0
    if max_val > 1e12:
        return 1.0 / 1000.0
    return 1.0 / 1000.0


def _plot_book_with_events(
    df_strategy: pd.DataFrame,
    df_fills: pd.DataFrame,
    df_cancels: pd.DataFrame,
    df_limits: pd.DataFrame,
    *,
    side: str,
    base_time: pd.Timestamp,
    width: int,
    height: int,
    elapsed_col: str = "elapsed_s",
    title_suffix: str = "",
) -> go.Figure:
    side_key = side.lower()
    bid_col = f"{side_key}_bid"
    ask_col = f"{side_key}_ask"
    eff_bid_col = f"{side_key}_eff_bid"
    eff_ask_col = f"{side_key}_eff_ask"

    elapsed_book = df_strategy[elapsed_col]

    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )

    if bid_col not in df_strategy.columns and eff_bid_col in df_strategy.columns:
        bid_col = eff_bid_col
    if ask_col not in df_strategy.columns and eff_ask_col in df_strategy.columns:
        ask_col = eff_ask_col

    for col, name in [(bid_col, f"{side} bid"), (ask_col, f"{side} ask")]:
        if col in df_strategy.columns:
            fig.add_trace(
                go.Scatter(
                    x=elapsed_book,
                    y=pd.to_numeric(df_strategy[col], errors="coerce"),
                    mode="lines",
                    name=name,
                    customdata=line_meta,
                    hovertemplate=(
                        "line=%{customdata}<br>t=%{x}s<br>" + col + "=%{y}<extra></extra>"
                    ),
                )
            )

    # Execution markers (fills only)
    if {"side", "price", "elapsed_s"}.issubset(df_fills.columns):
        buy_points = df_fills[df_fills["side"] == "buy"]
        sell_points = df_fills[df_fills["side"] == "sell"]
    else:
        buy_points = pd.DataFrame()
        sell_points = pd.DataFrame()

    def _add_markers(df_points: pd.DataFrame, name: str, color: str, symbol: str) -> None:
        if df_points.empty:
            return
        if "__lines__" in df_points.columns:
            meta = df_points["__lines__"].fillna("").tolist()
            hover = "lines=%{customdata}<br>t=%{x}s<br>price=%{y}<extra></extra>"
        else:
            meta = (
                df_points.get("__line__", pd.Series([np.nan] * len(df_points)))
                .astype("Int64")
                .tolist()
            )
            hover = "line=%{customdata}<br>t=%{x}s<br>price=%{y}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=df_points["elapsed_s"],
                y=df_points["price"],
                mode="markers",
                marker=dict(color=color, symbol=symbol, size=10),
                name=name,
                customdata=meta,
                hovertemplate=hover,
            )
        )

    _add_markers(buy_points, f"{side} buy fills", "green", "circle")
    _add_markers(sell_points, f"{side} sell fills", "red", "circle")

    if {"price", "elapsed_s"}.issubset(df_cancels.columns):
        _add_markers(df_cancels, f"{side} cancels", "gray", "x")

    def _add_limit_lines(df_lines: pd.DataFrame, name: str, color: str) -> None:
        if df_lines.empty or not {"start_s", "end_s", "price"}.issubset(df_lines.columns):
            return
        clean = df_lines.dropna(subset=["start_s", "end_s", "price"])
        if clean.empty:
            return
        xs: list[float | None] = []
        ys: list[float | None] = []
        meta: list[int | None] = []
        for _, row in clean.iterrows():
            xs.extend([row["start_s"], row["end_s"], None])
            ys.extend([row["price"], row["price"], None])
            lines_val = row.get("__lines__")
            if lines_val is None:
                lines_val = f"start={row.get('__line_start__')},end={row.get('__line_end__')}"
            meta.extend([lines_val, lines_val, None])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=name,
                line=dict(color=color, dash="dash"),
                customdata=meta,
                hovertemplate="lines=%{customdata}<br>t=%{x}s<br>limit=%{y}<extra></extra>",
            )
        )

    if {"side", "start_s", "end_s", "price"}.issubset(df_limits.columns):
        _add_limit_lines(df_limits[df_limits["side"] == "buy"], f"{side} buy limits", "green")
        _add_limit_lines(df_limits[df_limits["side"] == "sell"], f"{side} sell limits", "red")

    fig.update_layout(
        title=f"{side} orderbook + fills + cancels + limit orders{title_suffix}",
        xaxis_title="seconds",
        yaxis_title="price",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=5)
    fig.update_yaxes(showgrid=True, range=[0, 1], dtick=0.1, tickformat=".2f")
    return fig


def _plot_telemetry_book(
    df_telemetry: pd.DataFrame,
    *,
    side: str,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str = "elapsed_s",
    title_suffix: str = "",
) -> go.Figure:
    side_key = side.lower()
    elapsed = df_telemetry[elapsed_col]
    fig = go.Figure()
    line_meta = (
        df_telemetry.get("__line__", pd.Series([np.nan] * len(df_telemetry)))
        .astype("Int64")
        .tolist()
    )

    for col, name in [
        (f"{side_key}_l1.bid_p", f"telemetry {side_key}_bid"),
        (f"{side_key}_l1.ask_p", f"telemetry {side_key}_ask"),
    ]:
        if col not in df_telemetry.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=pd.to_numeric(df_telemetry[col], errors="coerce"),
                mode="lines",
                name=name,
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>" + col + "=%{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Telemetry {side} orderbook (bid/ask){title_suffix}",
        xaxis_title="seconds",
        yaxis_title="price",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True, range=[0, 1], dtick=0.1, tickformat=".2f")
    return fig


def _plot_btc_prices(
    df_strategy: pd.DataFrame,
    *,
    base_time: pd.Timestamp,
    width: int,
    height: int,
    x_tick: int,
    elapsed_override: pd.Series | None = None,
    title_suffix: str = "",
) -> go.Figure:
    if elapsed_override is None:
        btc_ts = _btc_timestamp(df_strategy)
        btc_time = _safe_ms_to_datetime(pd.to_numeric(btc_ts, errors="coerce"))
        elapsed = (btc_time - base_time).dt.total_seconds()
    else:
        elapsed = elapsed_override

    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )

    for col in ("btc_price", "price_to_beat", "coinbase_price"):
        if col not in df_strategy.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=pd.to_numeric(df_strategy[col], errors="coerce"),
                mode="lines",
                name=col,
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>" + col + "=%{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"BTC price vs strike{title_suffix}",
        xaxis_title="seconds",
        yaxis_title="price",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    return fig


def _plot_btc_deltas(
    df_strategy: pd.DataFrame,
    *,
    base_time: pd.Timestamp,
    width: int,
    height: int,
    x_tick: int,
    elapsed_override: pd.Series | None = None,
    title_suffix: str = "",
) -> go.Figure:
    if elapsed_override is None:
        btc_ts = _btc_timestamp(df_strategy)
        btc_time = _safe_ms_to_datetime(pd.to_numeric(btc_ts, errors="coerce"))
        elapsed = (btc_time - base_time).dt.total_seconds()
    else:
        elapsed = elapsed_override

    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )

    for col in ("btc_delta", "coinbase_delta", "delta"):
        if col not in df_strategy.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=pd.to_numeric(df_strategy[col], errors="coerce"),
                mode="lines",
                name=col,
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>" + col + "=%{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"BTC delta vs strike{title_suffix}",
        xaxis_title="seconds",
        yaxis_title="delta",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    return fig


def _plot_portfolio(
    df_portfolio: pd.DataFrame,
    *,
    base_time: pd.Timestamp,
    width: int,
    height: int,
    x_tick: int,
) -> go.Figure:
    ts = pd.to_numeric(df_portfolio.get("ts_ms"), errors="coerce")
    dt = _safe_ms_to_datetime(ts)
    elapsed = (dt - base_time).dt.total_seconds()

    fig = go.Figure()
    line_meta = (
        df_portfolio.get("__line__", pd.Series([np.nan] * len(df_portfolio)))
        .astype("Int64")
        .tolist()
    )

    for col in ("cash", "portfolio_mtm"):
        if col in df_portfolio.columns:
            fig.add_trace(
                go.Scatter(
                    x=elapsed,
                    y=pd.to_numeric(df_portfolio[col], errors="coerce"),
                    mode="lines",
                    name=col,
                    customdata=line_meta,
                    hovertemplate=(
                        "line=%{customdata}<br>t=%{x}s<br>" + col + "=%{y}<extra></extra>"
                    ),
                )
            )

    if "pnl" in df_portfolio.columns:
        pnl_series = pd.to_numeric(df_portfolio["pnl"], errors="coerce")
        pnl_mask = pnl_series.notna()
        if pnl_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=elapsed[pnl_mask],
                    y=pnl_series[pnl_mask],
                    mode="markers",
                    marker=dict(color="purple", symbol="diamond", size=10),
                    name="pnl",
                    customdata=list(pd.Series(line_meta)[pnl_mask]),
                    hovertemplate="line=%{customdata}<br>t=%{x}s<br>pnl=%{y}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Portfolio: cash / mtm / pnl",
        xaxis_title="seconds",
        yaxis_title="value",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    return fig


def _get_series(df: pd.DataFrame, *cols: str) -> pd.Series | None:
    for col in cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def _token_mid_series(df: pd.DataFrame, side_key: str) -> pd.Series | None:
    for col in (f"{side_key}_mid", f"{side_key}_eff_mid", f"{side_key}_l1.mid"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    if f"{side_key}_eff_bid" in df.columns and f"{side_key}_eff_ask" in df.columns:
        bid = pd.to_numeric(df[f"{side_key}_eff_bid"], errors="coerce")
        ask = pd.to_numeric(df[f"{side_key}_eff_ask"], errors="coerce")
        return 0.5 * (bid + ask)
    if f"{side_key}_bid" in df.columns and f"{side_key}_ask" in df.columns:
        bid = pd.to_numeric(df[f"{side_key}_bid"], errors="coerce")
        ask = pd.to_numeric(df[f"{side_key}_ask"], errors="coerce")
        return 0.5 * (bid + ask)
    fair_mid = f"fair_token_mid_{side_key}"
    if fair_mid in df.columns:
        return pd.to_numeric(df[fair_mid], errors="coerce")
    return None


def _plot_fair_vs_token_mid(
    df_strategy: pd.DataFrame,
    *,
    side: str,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
    title_suffix: str = "",
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    side_key = side.lower()
    fair = _get_series(df_strategy, f"fair_price_{side_key}")
    token_mid = _token_mid_series(df_strategy, side_key)
    if fair is None and token_mid is None:
        return None

    elapsed = pd.to_numeric(df_strategy[elapsed_col], errors="coerce")
    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )

    if token_mid is not None:
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=token_mid,
                mode="lines",
                name=f"{side} token mid",
                customdata=line_meta,
                hovertemplate="line=%{customdata}<br>t=%{x}s<br>token_mid=%{y}<extra></extra>",
            )
        )
    if fair is not None:
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=fair,
                mode="lines",
                name=f"{side} fair price",
                customdata=line_meta,
                hovertemplate="line=%{customdata}<br>t=%{x}s<br>fair=%{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Fair vs token mid ({side}){title_suffix}",
        xaxis_title="seconds",
        yaxis_title="price",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True, range=[0, 1], dtick=0.1, tickformat=".2f")
    return fig


def _plot_btc_vs_fair_delta(
    df_strategy: pd.DataFrame,
    *,
    side: str,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
    title_suffix: str = "",
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    side_key = side.lower()
    btc = _get_series(df_strategy, "btc_price", "btc.price")
    anchor_btc = _get_series(df_strategy, f"fair_anchor_btc_{side_key}")
    anchor_token_mid = _get_series(df_strategy, f"fair_anchor_token_mid_{side_key}")
    fair_price = _get_series(df_strategy, f"fair_price_{side_key}")

    delta_btc = None
    if btc is not None and anchor_btc is not None:
        delta_btc = btc - anchor_btc
    fair_delta = None
    if fair_price is not None and anchor_token_mid is not None:
        fair_delta = fair_price - anchor_token_mid

    if delta_btc is None and fair_delta is None:
        return None

    elapsed = pd.to_numeric(df_strategy[elapsed_col], errors="coerce")
    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )

    if delta_btc is not None:
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=delta_btc,
                mode="lines",
                name=f"{side} Δbtc(anchor)",
                customdata=line_meta,
                hovertemplate="line=%{customdata}<br>t=%{x}s<br>Δbtc=%{y}<extra></extra>",
            )
        )
    if fair_delta is not None:
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=fair_delta,
                mode="lines",
                name=f"{side} fair - anchor_token_mid",
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>fair_delta=%{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"BTC vs fair delta ({side}){title_suffix}",
        xaxis_title="seconds",
        yaxis_title="delta",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    return fig


def _plot_fair_vs_k_delta(
    df_strategy: pd.DataFrame,
    *,
    side: str,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
    title_suffix: str = "",
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    side_key = side.lower()
    fair_price = _get_series(df_strategy, f"fair_price_{side_key}")
    k = _get_series(df_strategy, "price_to_beat", "btc.price_to_beat")
    if fair_price is None or k is None:
        return None
    delta = fair_price - k
    elapsed = pd.to_numeric(df_strategy[elapsed_col], errors="coerce")
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=elapsed,
            y=delta,
            mode="lines",
            name=f"{side} fair - K",
            customdata=line_meta,
            hovertemplate="line=%{customdata}<br>t=%{x}s<br>fair-K=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Fair vs price_to_beat delta ({side}){title_suffix}",
        xaxis_title="seconds",
        yaxis_title="delta",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    return fig


def _plot_anchor_drift(
    df_strategy: pd.DataFrame,
    *,
    side: str,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
    title_suffix: str = "",
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    side_key = side.lower()
    anchor_token_mid = _get_series(df_strategy, f"fair_anchor_token_mid_{side_key}")
    anchor_btc = _get_series(df_strategy, f"fair_anchor_btc_{side_key}")
    if anchor_token_mid is None and anchor_btc is None:
        return None

    elapsed = pd.to_numeric(df_strategy[elapsed_col], errors="coerce")
    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )

    if anchor_token_mid is not None:
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=anchor_token_mid,
                mode="lines",
                name=f"{side} anchor token mid",
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>anchor_token_mid=%{y}<extra></extra>"
                ),
            )
        )
    if anchor_btc is not None:
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=anchor_btc,
                mode="lines",
                name=f"{side} anchor btc",
                yaxis="y2",
                customdata=line_meta,
                hovertemplate="line=%{customdata}<br>t=%{x}s<br>anchor_btc=%{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Anchor drift ({side}){title_suffix}",
        xaxis_title="seconds",
        yaxis_title="anchor token mid",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
        yaxis2=dict(
            title="anchor btc",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    return fig


def _plot_up_book_react(
    df_strategy: pd.DataFrame,
    *,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str = "elapsed_s",
    title_suffix: str = "",
) -> go.Figure:
    elapsed = df_strategy[elapsed_col]
    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )

    for col, name in [("up_bid", "up_bid"), ("up_ask", "up_ask")]:
        if col not in df_strategy.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=pd.to_numeric(df_strategy[col], errors="coerce"),
                mode="lines",
                name=name,
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>" + col + "=%{y}<extra></extra>"
                ),
            )
        )

    if "p_react_up" in df_strategy.columns:
        fig.add_trace(
            go.Scatter(
                x=elapsed,
                y=pd.to_numeric(df_strategy["p_react_up"], errors="coerce"),
                mode="lines",
                name="p_react_up",
                yaxis="y2",
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>p_react_up=%{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Up book + p_react_up{title_suffix}",
        xaxis_title="seconds",
        yaxis_title="price",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
        yaxis2=dict(
            title="p_react_up",
            overlaying="y",
            side="right",
            range=[0, 1],
            showgrid=False,
        ),
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True, range=[0, 1], dtick=0.1, tickformat=".2f")
    return fig


def _plot_price_sources_delta_orders(
    df_telemetry: pd.DataFrame,
    source_cols: list[str],
    primary_col: str | None,
    df_fills: pd.DataFrame,
    df_cancels: pd.DataFrame,
    *,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
    title: str,
) -> go.Figure | None:
    if df_telemetry is None or df_telemetry.empty:
        return None
    if not source_cols:
        return None
    k_col = "btc.price_to_beat" if "btc.price_to_beat" in df_telemetry.columns else None
    if k_col is None:
        return None
    if elapsed_col not in df_telemetry.columns:
        return None

    x = pd.to_numeric(df_telemetry[elapsed_col], errors="coerce")
    k = pd.to_numeric(df_telemetry[k_col], errors="coerce")

    fig = go.Figure()
    line_meta = (
        df_telemetry.get("__line__", pd.Series([np.nan] * len(df_telemetry)))
        .astype("Int64")
        .tolist()
    )

    for col in source_cols:
        if col not in df_telemetry.columns:
            continue
        y = pd.to_numeric(df_telemetry[col], errors="coerce") - k
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=col,
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>" + col + "=%{y}<extra></extra>"
                ),
            )
        )

    primary_delta = None
    if primary_col and primary_col in df_telemetry.columns:
        primary_delta = pd.to_numeric(df_telemetry[primary_col], errors="coerce") - k

    def _map_to_nearest(tele_x: np.ndarray, tele_y: np.ndarray, pts: np.ndarray) -> np.ndarray:
        mask = np.isfinite(tele_x) & np.isfinite(tele_y)
        tele_x = tele_x[mask]
        tele_y = tele_y[mask]
        if tele_x.size == 0:
            return np.full_like(pts, np.nan, dtype=float)
        order = np.argsort(tele_x)
        tele_x = tele_x[order]
        tele_y = tele_y[order]
        pts = pts.astype(float)
        idx = np.searchsorted(tele_x, pts, side="left")
        if tele_x.size == 1:
            return np.full_like(pts, tele_y[0], dtype=float)
        idx = np.clip(idx, 1, tele_x.size - 1)
        left = tele_x[idx - 1]
        right = tele_x[idx]
        choose_right = (pts - left) > (right - pts)
        sel = np.where(choose_right, idx, idx - 1)
        return tele_y[sel]

    def _add_markers(df_points: pd.DataFrame, name: str, color: str, symbol: str) -> None:
        if df_points.empty:
            return
        pts_x = pd.to_numeric(df_points["elapsed_s"], errors="coerce").to_numpy()
        if primary_delta is None:
            pts_y = np.zeros_like(pts_x, dtype=float)
        else:
            pts_y = _map_to_nearest(
                x.to_numpy(dtype=float),
                primary_delta.to_numpy(dtype=float),
                pts_x.astype(float),
            )
        df_plot = df_points.copy()
        df_plot["price"] = pts_y
        df_plot = df_plot.dropna(subset=["price", "elapsed_s"])
        if df_plot.empty:
            return
        meta = (
            df_plot.get("__lines__", pd.Series([np.nan] * len(df_plot)))
            .fillna("")
            .tolist()
        )
        fig.add_trace(
            go.Scatter(
                x=df_plot["elapsed_s"],
                y=df_plot["price"],
                mode="markers",
                marker=dict(color=color, symbol=symbol, size=10),
                name=name,
                customdata=meta,
                hovertemplate="lines=%{customdata}<br>t=%{x}s<br>delta=%{y}<extra></extra>",
            )
        )

    if not df_fills.empty and "side" in df_fills.columns:
        _add_markers(df_fills[df_fills["side"] == "buy"], "buy fills", "green", "circle")
        _add_markers(df_fills[df_fills["side"] == "sell"], "sell fills", "red", "circle")
    if not df_cancels.empty:
        _add_markers(df_cancels, "cancels", "gray", "x")

    fig.update_layout(
        title=title,
        xaxis_title="seconds",
        yaxis_title="price - K",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    return fig


def _plot_ab_fit(
    df_strategy: pd.DataFrame,
    *,
    width: int,
    height: int,
    x_tick: int,
    window: int,
    stride: int,
) -> go.Figure | None:
    required = {"delta", "scale", "up_bid", "up_ask", "down_bid", "down_ask", "elapsed_s"}
    if not required.issubset(df_strategy.columns):
        return None

    delta = pd.to_numeric(df_strategy["delta"], errors="coerce")
    scale = pd.to_numeric(df_strategy["scale"], errors="coerce")
    up_bid = pd.to_numeric(df_strategy["up_bid"], errors="coerce")
    up_ask = pd.to_numeric(df_strategy["up_ask"], errors="coerce")
    down_bid = pd.to_numeric(df_strategy["down_bid"], errors="coerce")
    down_ask = pd.to_numeric(df_strategy["down_ask"], errors="coerce")
    elapsed = pd.to_numeric(df_strategy["elapsed_s"], errors="coerce")

    up_mid = 0.5 * (up_bid + up_ask)
    down_mid = 0.5 * (down_bid + down_ask)
    y = 0.5 * (up_mid + (1 - down_mid))
    z = delta / (scale + 1e-12)

    mask = (
        delta.notna()
        & scale.notna()
        & up_bid.notna()
        & up_ask.notna()
        & down_bid.notna()
        & down_ask.notna()
        & elapsed.notna()
    )
    y = y[mask].to_numpy()
    z = z[mask].to_numpy()
    t = elapsed[mask].to_numpy()

    if len(z) < max(window, 5):
        return None

    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    ys = _logit(y)

    out_t = []
    out_a = []
    out_b = []
    for i in range(window - 1, len(z), max(1, stride)):
        z_win = z[i - window + 1 : i + 1]
        y_win = ys[i - window + 1 : i + 1]
        var_z = np.var(z_win)
        if var_z == 0:
            continue
        b = np.cov(z_win, y_win, bias=True)[0, 1] / var_z
        a = y_win.mean() - b * z_win.mean()
        out_t.append(t[i])
        out_a.append(a)
        out_b.append(b)

    if not out_t:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=out_t,
            y=out_a,
            mode="lines",
            name="a",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=out_t,
            y=out_b,
            mode="lines",
            name="b",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Rolling logit fit (a,b) from strategy",
        xaxis_title="seconds",
        yaxis_title="a",
        hovermode="x unified",
        height=height,
        width=width,
        autosize=False,
        yaxis2=dict(
            title="b",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )
    fig.update_xaxes(showgrid=True, dtick=x_tick)
    fig.update_yaxes(showgrid=True)
    return fig


def _parse_execution_events(
    exec_records: list[dict],
    side: str,
    base_time: pd.Timestamp | float | None,
    *,
    end_time_s: float | None = None,
    ts_field: str = "ts_ms",
    use_raw_time: bool = False,
    time_scale: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    side_norm = side.lower()

    def _to_int(val) -> int | None:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _norm_outcome(val) -> str | None:
        if val is None:
            return None
        txt = str(val).strip().lower()
        if txt in {"yes", "y", "true", "1"}:
            return "up"
        if txt in {"no", "n", "false", "0"}:
            return "down"
        return txt

    def _norm_side(val) -> str | None:
        if val is None:
            return None
        txt = str(val).strip().lower()
        if txt in {"buy", "bid"}:
            return "buy"
        if txt in {"sell", "ask"}:
            return "sell"
        return txt

    def _event_ts_ms(rec: dict) -> int | None:
        return rec.get(ts_field)

    def _elapsed_from_ts(ts_val: int | None) -> float | None:
        if ts_val is None or base_time is None:
            return None
        if use_raw_time:
            if time_scale is None:
                return None
            return (float(ts_val) - float(base_time)) * time_scale
        dt = _safe_ms_to_datetime(pd.Series([ts_val])).iloc[0]
        if pd.isna(dt):
            return None
        return (dt - base_time).total_seconds()

    orders: dict[tuple, dict] = {}

    for rec in exec_records:
        event_kind = rec.get("event_kind")
        event = rec.get("event")
        if event_kind is not None:
            event_kind = str(event_kind)
        if event is not None:
            event = str(event)

        client_id = _to_int(rec.get("client_order_id"))
        strat_id = _to_int(rec.get("strategy_order_id"))
        market_id = rec.get("market_id")
        order_id = client_id if client_id is not None else strat_id
        if order_id is None:
            continue
        key = (order_id, market_id)

        ts_val = _event_ts_ms(rec)
        side_val = _norm_side(rec.get("side"))
        outcome_val = _norm_outcome(rec.get("outcome"))
        order_type = (rec.get("order_type") or "").strip().lower() or None
        limit_price = rec.get("limit_price")
        avg_price = rec.get("avg_price")
        status = str(rec.get("status") or "").strip().lower()
        decision_id = rec.get("decision_id")
        filled_amount = rec.get("filled_amount")

        order = orders.setdefault(
            key,
            {
                "created_ts": None,
                "last_ts": None,
                "side": None,
                "outcome": None,
                "order_type": None,
                "limit_price": None,
                "decision_id": None,
                "status_final": None,
                "filled_amount": None,
                "avg_price": None,
                "filled_ts": None,
                "canceled_ts": None,
                "created_line": None,
                "filled_line": None,
                "canceled_line": None,
                "lines": set(),
            },
        )
        line_no = rec.get("__line__")
        if line_no is not None:
            try:
                order["lines"].add(int(line_no))
            except (TypeError, ValueError):
                pass

        if ts_val is not None:
            order["last_ts"] = ts_val
            if order["created_ts"] is None and event_kind == "request":
                order["created_ts"] = ts_val
                order["created_line"] = line_no
            if order["created_ts"] is None and event_kind == "report" and status in {"submitted", "accepted"}:
                order["created_ts"] = ts_val
                order["created_line"] = line_no

        if side_val:
            order["side"] = side_val
        if outcome_val:
            order["outcome"] = outcome_val
        if order_type:
            order["order_type"] = order_type
        if limit_price is not None:
            order["limit_price"] = limit_price
        if decision_id is not None:
            order["decision_id"] = decision_id
        if filled_amount is not None:
            order["filled_amount"] = filled_amount
        if avg_price is not None:
            order["avg_price"] = avg_price

        if event_kind == "fill_applied":
            order["status_final"] = "filled"
            if ts_val is not None:
                order["filled_ts"] = ts_val
                order["filled_line"] = line_no
            if avg_price is not None:
                order["avg_price"] = avg_price

        if event_kind == "report":
            if status == "filled":
                order["status_final"] = "filled"
                if ts_val is not None:
                    order["filled_ts"] = ts_val
                    order["filled_line"] = line_no
                if avg_price is not None:
                    order["avg_price"] = avg_price
            elif status == "canceled":
                order["status_final"] = "canceled"
                if ts_val is not None:
                    order["canceled_ts"] = ts_val
                    order["canceled_line"] = line_no

    fill_rows: list[dict] = []
    cancel_rows: list[dict] = []
    limit_rows: list[dict] = []

    for order in orders.values():
        outcome_val = order.get("outcome")
        if outcome_val is None or outcome_val != side_norm:
            continue

        side_val = order.get("side")
        order_type = order.get("order_type")
        limit_price = order.get("limit_price")
        avg_price = order.get("avg_price")
        status_final = order.get("status_final") or "open"
        lines = order.get("lines") or set()
        lines_str = ",".join(str(v) for v in sorted(lines)) if lines else None

        created_ts = order.get("created_ts") or order.get("last_ts")
        filled_ts = order.get("filled_ts")
        canceled_ts = order.get("canceled_ts")

        if status_final == "filled":
            ts_val = filled_ts or order.get("last_ts")
            elapsed = _elapsed_from_ts(ts_val)
            price = avg_price if avg_price is not None else limit_price
            if elapsed is not None and price is not None:
                fill_rows.append(
                    {
                        "side": side_val,
                        "price": price,
                        "elapsed_s": elapsed,
                        "__line__": order.get("filled_line"),
                        "__lines__": lines_str,
                    }
                )

        if status_final == "canceled":
            ts_val = canceled_ts or order.get("last_ts")
            elapsed = _elapsed_from_ts(ts_val)
            if elapsed is not None and limit_price is not None:
                cancel_rows.append(
                    {
                        "price": limit_price,
                        "elapsed_s": elapsed,
                        "__line__": order.get("canceled_line"),
                        "__lines__": lines_str,
                    }
                )

        if order_type in {"gtc", "gtd"} and created_ts is not None:
            start_s = _elapsed_from_ts(created_ts)
            end_ts = filled_ts or canceled_ts
            end_s = _elapsed_from_ts(end_ts) if end_ts is not None else end_time_s
            if start_s is not None and end_s is not None and limit_price is not None:
                limit_rows.append(
                    {
                        "side": side_val,
                        "price": limit_price,
                        "start_s": start_s,
                        "end_s": end_s,
                        "__line_start__": order.get("created_line"),
                        "__line_end__": order.get("filled_line") or order.get("canceled_line"),
                        "__lines__": lines_str,
                    }
                )

    fills_df = (
        pd.DataFrame(fill_rows)
        if fill_rows
        else pd.DataFrame(columns=["side", "price", "elapsed_s", "__line__", "__lines__"])
    )
    cancels_df = (
        pd.DataFrame(cancel_rows)
        if cancel_rows
        else pd.DataFrame(columns=["price", "elapsed_s", "__line__", "__lines__"])
    )
    limits_df = (
        pd.DataFrame(limit_rows)
        if limit_rows
        else pd.DataFrame(
            columns=["side", "price", "start_s", "end_s", "__line_start__", "__line_end__", "__lines__"]
        )
    )
    return fills_df, cancels_df, limits_df


st.set_page_config(page_title="Market Combo Dashboard", layout="wide")

st.title("Market Combo Dashboard (Strategy + Execution)")

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
    market_dir = st.text_input(
        "Market directory",
        value="",
        placeholder="/path/to/output/json/markets/<slug>",
    )
    strategy_name = st.text_input("Strategy filter (optional)", value="")
    sides = st.multiselect("Sides", options=["Up", "Down"], default=["Up", "Down"])
    downsample = st.text_input("Downsample rule (strategy)", value="50ms")
    if downsample.strip().lower() == "none":
        downsample = None
    max_points = st.number_input("Max points", value=500000, step=50000)
    st.header("Display")
    x_tick = st.number_input("X tick (seconds)", value=5, step=1)
    height = st.number_input("Chart height", value=500, step=50)
    base_width = st.number_input("Base chart width (px)", value=1200, step=100)
    xscale = st.number_input("X scale", value=1.0, step=0.25)
    st.header("AB fit")
    ab_window = st.number_input("AB window (points)", value=500, step=50)
    ab_stride = st.number_input("AB stride (points)", value=25, step=5)

if not market_dir:
    st.info("Set market directory in the sidebar.")
    st.stop()

if not sides:
    st.info("Select at least one side (Up/Down) in the sidebar.")
    st.stop()

market_path = Path(market_dir)
strategy_path = market_path / "strategy.jsonl"
execution_path = market_path / "execution.jsonl"
telemetry_path = market_path / "telemetry.jsonl"
portfolio_path = market_path / "portfolio.jsonl"

if not strategy_path.exists():
    st.error(f"Missing strategy.jsonl: {strategy_path}")
    st.stop()
if not execution_path.exists():
    st.error(f"Missing execution.jsonl: {execution_path}")
    st.stop()

with st.spinner("Loading strategy..."):
    strategy_records = load_jsonl(strategy_path)
    if strategy_name:
        strategy_records = [r for r in strategy_records if r.get("strategy") == strategy_name]
    df_strategy = pd.json_normalize(strategy_records, sep=".")
    if df_strategy.empty:
        st.error("No strategy records after filtering.")
        st.stop()
    if "decision_ts_ms" in df_strategy.columns and pd.to_numeric(
        df_strategy["decision_ts_ms"], errors="coerce"
    ).notna().any():
        book_ts_ms = df_strategy["decision_ts_ms"]
    else:
        book_ts_ms = _book_timestamp(df_strategy)
    if book_ts_ms is None:
        st.error("No book timestamp found in strategy.jsonl")
        st.stop()
    df_strategy["timestamp"] = _safe_ms_to_datetime(pd.to_numeric(book_ts_ms, errors="coerce"))
    df_strategy = maybe_downsample(df_strategy, rule=downsample, max_points=int(max_points))
    base_time = None

df_telemetry = None
if telemetry_path.exists():
    with st.spinner("Loading telemetry..."):
        telemetry_records = load_jsonl(telemetry_path)
        df_telemetry = pd.json_normalize(telemetry_records, sep=".")
        if not df_telemetry.empty:
            tele_ts_ms = _telemetry_book_timestamp(df_telemetry, side="book")
            df_telemetry["timestamp"] = _safe_ms_to_datetime(
                pd.to_numeric(tele_ts_ms, errors="coerce")
            )
            df_telemetry = maybe_downsample(
                df_telemetry, rule=downsample, max_points=int(max_points)
            )

strategy_min = (
    df_strategy["timestamp"].min()
    if "timestamp" in df_strategy.columns and df_strategy["timestamp"].notna().any()
    else None
)
telemetry_min = (
    df_telemetry["timestamp"].min()
    if df_telemetry is not None
    and "timestamp" in df_telemetry.columns
    and df_telemetry["timestamp"].notna().any()
    else None
)
base_candidates = [ts for ts in [strategy_min, telemetry_min] if ts is not None and pd.notna(ts)]
if base_candidates:
    base_time = min(base_candidates)
else:
    st.error("No valid timestamps found in strategy or telemetry.")
    st.stop()

df_strategy["elapsed_s"] = (df_strategy["timestamp"] - base_time).dt.total_seconds()
if df_telemetry is not None and "timestamp" in df_telemetry.columns:
    df_telemetry["elapsed_s"] = (df_telemetry["timestamp"] - base_time).dt.total_seconds()

with st.spinner("Loading execution..."):
    exec_records = load_jsonl(execution_path)
    end_time_s = float(df_strategy["elapsed_s"].max())
    df_exec_up, df_cancels_up, df_limits_up = _parse_execution_events(
        exec_records, side="Up", base_time=base_time, end_time_s=end_time_s
    )
    df_exec_down, df_cancels_down, df_limits_down = _parse_execution_events(
        exec_records, side="Down", base_time=base_time, end_time_s=end_time_s
    )

df_portfolio = None
if portfolio_path.exists():
    with st.spinner("Loading portfolio..."):
        portfolio_records = load_jsonl(portfolio_path)
        df_portfolio = pd.json_normalize(portfolio_records, sep=".")
        if df_portfolio.empty:
            df_portfolio = None

width = int(base_width * xscale)

if df_telemetry is not None and not df_telemetry.empty:
    for s in sides:
        tele_fig = _plot_telemetry_book(
            df_telemetry, side=s, width=width, height=int(height), x_tick=int(x_tick)
        )
        st.plotly_chart(tele_fig, width=width, config={"responsive": False})
else:
    st.info("Telemetry orderbook not available (telemetry.jsonl missing or empty).")

for s in sides:
    df_exec_side = df_exec_up if s.lower() == "up" else df_exec_down
    df_cancels_side = df_cancels_up if s.lower() == "up" else df_cancels_down
    df_limits_side = df_limits_up if s.lower() == "up" else df_limits_down
    fig = _plot_book_with_events(
        df_strategy,
        df_exec_side,
        df_cancels_side,
        df_limits_side,
        side=s,
        base_time=base_time,
        width=width,
        height=int(height),
    )
    fig.update_xaxes(dtick=int(x_tick))
    st.plotly_chart(fig, width=width, config={"responsive": False})

if "p_react_up" in df_strategy.columns:
    react_fig = _plot_up_book_react(
        df_strategy,
        width=width,
        height=int(height),
        x_tick=int(x_tick),
        elapsed_col="elapsed_s",
        title_suffix=" (strategy)",
    )
    st.plotly_chart(react_fig, width=width, config={"responsive": False})

st.subheader("Up orderbook + orders (engine time)")
engine_series = _strategy_engine_series(df_strategy)
tele_engine_series = (
    pd.to_numeric(df_telemetry.get("engine_ts_ms"), errors="coerce")
    if df_telemetry is not None and "engine_ts_ms" in df_telemetry.columns
    else pd.Series(dtype="float64")
)
exec_engine_series = pd.to_numeric(
    pd.Series(
        [rec.get("engine_ts_ms") for rec in exec_records if rec.get("engine_ts_ms") is not None]
    ),
    errors="coerce",
)
engine_candidates = []
if not engine_series.empty and engine_series.notna().any():
    engine_candidates.append(engine_series.min())
if not tele_engine_series.empty and tele_engine_series.notna().any():
    engine_candidates.append(tele_engine_series.min())
if not exec_engine_series.empty and exec_engine_series.notna().any():
    engine_candidates.append(exec_engine_series.min())

if engine_candidates:
    engine_base_raw = min(engine_candidates)
    ref_span_ms = float(df_strategy["elapsed_s"].max()) * 1000.0 if "elapsed_s" in df_strategy.columns else None
    if not engine_series.dropna().empty:
        scale_source = engine_series
    elif not tele_engine_series.dropna().empty:
        scale_source = tele_engine_series
    else:
        scale_source = exec_engine_series
    engine_scale = _infer_engine_scale_seconds(scale_source, ref_span_ms)

    if not engine_series.empty and engine_series.notna().any():
        df_strategy["elapsed_engine_s"] = (engine_series - engine_base_raw) * engine_scale
    if df_telemetry is not None and "engine_ts_ms" in df_telemetry.columns:
        df_telemetry["elapsed_engine_s"] = (tele_engine_series - engine_base_raw) * engine_scale

    end_time_engine_s = None
    if "elapsed_engine_s" in df_strategy.columns and df_strategy["elapsed_engine_s"].notna().any():
        end_time_engine_s = df_strategy["elapsed_engine_s"].max()
    elif df_telemetry is not None and "elapsed_engine_s" in df_telemetry.columns:
        end_time_engine_s = df_telemetry["elapsed_engine_s"].max()
    elif not exec_engine_series.empty and exec_engine_series.notna().any():
        end_time_engine_s = (exec_engine_series.max() - engine_base_raw) * engine_scale

    df_exec_engine_up, df_cancels_engine_up, df_limits_engine_up = _parse_execution_events(
        exec_records,
        side="Up",
        base_time=engine_base_raw,
        end_time_s=end_time_engine_s,
        ts_field="engine_ts_ms",
        use_raw_time=True,
        time_scale=engine_scale,
    )
    df_exec_engine_down, df_cancels_engine_down, df_limits_engine_down = _parse_execution_events(
        exec_records,
        side="Down",
        base_time=engine_base_raw,
        end_time_s=end_time_engine_s,
        ts_field="engine_ts_ms",
        use_raw_time=True,
        time_scale=engine_scale,
    )

    if "elapsed_engine_s" in df_strategy.columns:
        engine_fig = _plot_book_with_events(
            df_strategy,
            df_exec_engine_up,
            df_cancels_engine_up,
            df_limits_engine_up,
            side="Up",
            base_time=base_time,
            width=width,
            height=int(height),
            elapsed_col="elapsed_engine_s",
            title_suffix=" (engine)",
        )
        engine_fig.update_xaxes(dtick=int(x_tick))
        st.plotly_chart(engine_fig, width=width, config={"responsive": False})

        engine_fig_down = _plot_book_with_events(
            df_strategy,
            df_exec_engine_down,
            df_cancels_engine_down,
            df_limits_engine_down,
            side="Down",
            base_time=base_time,
            width=width,
            height=int(height),
            elapsed_col="elapsed_engine_s",
            title_suffix=" (engine)",
        )
        engine_fig_down.update_xaxes(dtick=int(x_tick))
        st.plotly_chart(engine_fig_down, width=width, config={"responsive": False})
    else:
        st.info("strategy engine timestamps missing; skipping engine orderbook graphs.")

    if df_telemetry is not None and "elapsed_engine_s" in df_telemetry.columns:
        source_cols, primary_col = _telemetry_price_sources(df_telemetry)
        delta_fig_up = _plot_price_sources_delta_orders(
            df_telemetry,
            source_cols,
            primary_col,
            df_exec_engine_up,
            df_cancels_engine_up,
            width=width,
            height=int(height),
            x_tick=int(x_tick),
            elapsed_col="elapsed_engine_s",
            title="Price sources delta vs K + orders (Up, engine)",
        )
        if delta_fig_up is not None:
            st.plotly_chart(delta_fig_up, width=width, config={"responsive": False})

        delta_fig_down = _plot_price_sources_delta_orders(
            df_telemetry,
            source_cols,
            primary_col,
            df_exec_engine_down,
            df_cancels_engine_down,
            width=width,
            height=int(height),
            x_tick=int(x_tick),
            elapsed_col="elapsed_engine_s",
            title="Price sources delta vs K + orders (Down, engine)",
        )
        if delta_fig_down is not None:
            st.plotly_chart(delta_fig_down, width=width, config={"responsive": False})
    else:
        st.info("Telemetry engine_ts_ms missing; cannot plot price-source delta vs K with orders.")

    if "elapsed_engine_s" in df_strategy.columns and "p_react_up" in df_strategy.columns:
        react_fig = _plot_up_book_react(
            df_strategy,
            width=width,
            height=int(height),
            x_tick=int(x_tick),
            elapsed_col="elapsed_engine_s",
            title_suffix=" (engine)",
        )
        st.plotly_chart(react_fig, width=width, config={"responsive": False})

    if "elapsed_engine_s" in df_strategy.columns:
        btc_engine_fig = _plot_btc_prices(
            df_strategy,
            base_time=base_time,
            width=width,
            height=int(height),
            x_tick=int(x_tick),
            elapsed_override=df_strategy["elapsed_engine_s"],
            title_suffix=" (engine)",
        )
        st.plotly_chart(btc_engine_fig, width=width, config={"responsive": False})
else:
    st.info("engine timestamps missing in strategy/telemetry/execution; cannot draw engine-time charts.")

price_fig = _plot_btc_prices(
    df_strategy,
    base_time=base_time,
    width=width,
    height=int(height),
    x_tick=int(x_tick),
)
st.plotly_chart(price_fig, width=width, config={"responsive": False})

delta_fig = _plot_btc_deltas(
    df_strategy,
    base_time=base_time,
    width=width,
    height=int(height),
    x_tick=int(x_tick),
)
st.plotly_chart(delta_fig, width=width, config={"responsive": False})

ab_fig = _plot_ab_fit(
    df_strategy,
    width=width,
    height=int(height),
    x_tick=int(x_tick),
    window=int(ab_window),
    stride=int(ab_stride),
)
if ab_fig is not None:
    st.plotly_chart(ab_fig, width=width, config={"responsive": False})

if df_portfolio is not None:
    portfolio_fig = _plot_portfolio(
        df_portfolio,
        base_time=base_time,
        width=width,
        height=int(height),
        x_tick=int(x_tick),
    )
    st.plotly_chart(portfolio_fig, width=width, config={"responsive": False})

if "elapsed_engine_s" in df_strategy.columns and df_strategy["elapsed_engine_s"].notna().any():
    st.subheader("Fair / Anchor (engine time)")
    for side in sides:
        fair_mid_fig = _plot_fair_vs_token_mid(
            df_strategy,
            side=side,
            width=width,
            height=int(height),
            x_tick=int(x_tick),
            elapsed_col="elapsed_engine_s",
            title_suffix=" (engine)",
        )
        if fair_mid_fig is not None:
            st.plotly_chart(fair_mid_fig, width=width, config={"responsive": False})

        btc_fair_delta_fig = _plot_btc_vs_fair_delta(
            df_strategy,
            side=side,
            width=width,
            height=int(height),
            x_tick=int(x_tick),
            elapsed_col="elapsed_engine_s",
            title_suffix=" (engine)",
        )
        if btc_fair_delta_fig is not None:
            st.plotly_chart(btc_fair_delta_fig, width=width, config={"responsive": False})

        fair_k_fig = _plot_fair_vs_k_delta(
            df_strategy,
            side=side,
            width=width,
            height=int(height),
            x_tick=int(x_tick),
            elapsed_col="elapsed_engine_s",
            title_suffix=" (engine)",
        )
        if fair_k_fig is not None:
            st.plotly_chart(fair_k_fig, width=width, config={"responsive": False})

        anchor_fig = _plot_anchor_drift(
            df_strategy,
            side=side,
            width=width,
            height=int(height),
            x_tick=int(x_tick),
            elapsed_col="elapsed_engine_s",
            title_suffix=" (engine)",
        )
        if anchor_fig is not None:
            st.plotly_chart(anchor_fig, width=width, config={"responsive": False})
