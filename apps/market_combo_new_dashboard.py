from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting.io import load_jsonl
from plotting.processing import maybe_downsample, _safe_ms_to_datetime


def _strategy_timestamp(df: pd.DataFrame) -> pd.Series:
    for col in ("engine_ts_ms", "decision_ts_ms", "ts_ms"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                return series
    return pd.Series(dtype="float64")


def _elapsed_seconds(ts: pd.Series, base_time: pd.Timestamp) -> pd.Series:
    dt = _safe_ms_to_datetime(pd.to_numeric(ts, errors="coerce"))
    return (dt - base_time).dt.total_seconds()


def _plot_book_with_events(
    df_strategy: pd.DataFrame,
    df_fills: pd.DataFrame,
    df_cancels: pd.DataFrame,
    df_limits: pd.DataFrame,
    df_accepts: pd.DataFrame | None = None,
    df_rejects: pd.DataFrame | None = None,
    *,
    side: str,
    width: int,
    height: int,
    elapsed_col: str = "elapsed_s",
) -> go.Figure:
    side_key = side.lower()
    bid_col = f"{side_key}_bid"
    ask_col = f"{side_key}_ask"

    elapsed_book = df_strategy[elapsed_col]

    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )

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

    def _add_markers(df_points: pd.DataFrame, name: str, color: str, symbol: str) -> None:
        if df_points.empty:
            return
        meta = (
            df_points.get("__lines__", pd.Series([np.nan] * len(df_points)))
            .fillna("")
            .tolist()
        )
        fig.add_trace(
            go.Scatter(
                x=df_points["elapsed_s"],
                y=df_points["price"],
                mode="markers",
                marker=dict(color=color, symbol=symbol, size=10),
                name=name,
                customdata=meta,
                hovertemplate="lines=%{customdata}<br>t=%{x}s<br>price=%{y}<extra></extra>",
            )
        )

    if df_accepts is not None and {"side", "price", "elapsed_s"}.issubset(df_accepts.columns):
        _add_markers(
            df_accepts[df_accepts["side"] == "buy"],
            f"{side} buy accepted",
            "green",
            "circle-open",
        )
        _add_markers(
            df_accepts[df_accepts["side"] == "sell"],
            f"{side} sell accepted",
            "red",
            "circle-open",
        )

    if {"side", "price", "elapsed_s"}.issubset(df_fills.columns):
        _add_markers(df_fills[df_fills["side"] == "buy"], f"{side} buy fills", "green", "circle")
        _add_markers(df_fills[df_fills["side"] == "sell"], f"{side} sell fills", "red", "circle")

    if {"price", "elapsed_s"}.issubset(df_cancels.columns):
        _add_markers(df_cancels, f"{side} cancels", "gray", "x")

    if df_rejects is not None and {"price", "elapsed_s"}.issubset(df_rejects.columns):
        if "side" in df_rejects.columns:
            _add_markers(
                df_rejects[df_rejects["side"] == "buy"],
                f"{side} buy rejects",
                "orange",
                "x",
            )
            _add_markers(
                df_rejects[df_rejects["side"] == "sell"],
                f"{side} sell rejects",
                "purple",
                "x",
            )
        else:
            _add_markers(df_rejects, f"{side} rejects", "orange", "x")

    def _add_limit_lines(df_lines: pd.DataFrame, name: str, color: str) -> None:
        if df_lines.empty or not {"start_s", "end_s", "price"}.issubset(df_lines.columns):
            return
        clean = df_lines.dropna(subset=["start_s", "end_s", "price"])
        if clean.empty:
            return
        xs: list[float | None] = []
        ys: list[float | None] = []
        meta: list[str | None] = []
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
        title=f"{side} orderbook + fills + cancels + limit orders",
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


def _plot_price_vs_k(
    df_strategy: pd.DataFrame,
    *,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    cols = [c for c in ("btc_price", "price_to_beat", "binance_price") if c in df_strategy.columns]
    if not cols:
        return None
    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )
    for col in cols:
        fig.add_trace(
            go.Scatter(
                x=df_strategy[elapsed_col],
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
        title="BTC price vs strike",
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


def _plot_price_deltas(
    df_strategy: pd.DataFrame,
    *,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
    df_fills: pd.DataFrame | None = None,
    df_cancels: pd.DataFrame | None = None,
    df_rejects: pd.DataFrame | None = None,
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    if "price_to_beat" not in df_strategy.columns:
        return None

    k = pd.to_numeric(df_strategy["price_to_beat"], errors="coerce")
    series = []
    if "btc_price" in df_strategy.columns:
        series.append(("rtds_delta", pd.to_numeric(df_strategy["btc_price"], errors="coerce") - k))
    if "binance_price" in df_strategy.columns:
        series.append(
            ("binance_delta", pd.to_numeric(df_strategy["binance_price"], errors="coerce") - k)
        )

    if not series:
        return None

    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )
    for name, data in series:
        fig.add_trace(
            go.Scatter(
                x=df_strategy[elapsed_col],
                y=data,
                mode="lines",
                name=name,
                customdata=line_meta,
                hovertemplate=(
                    "line=%{customdata}<br>t=%{x}s<br>" + name + "=%{y}<extra></extra>"
                ),
            )
        )

    def _add_markers(df_points: pd.DataFrame, name: str, color: str, symbol: str) -> None:
        if df_points is None or df_points.empty:
            return
        meta = (
            df_points.get("__lines__", pd.Series([np.nan] * len(df_points)))
            .fillna("")
            .tolist()
        )
        fig.add_trace(
            go.Scatter(
                x=df_points["elapsed_s"],
                y=df_points["price"],
                mode="markers",
                marker=dict(color=color, symbol=symbol, size=9),
                name=name,
                customdata=meta,
                hovertemplate="lines=%{customdata}<br>t=%{x}s<br>delta=%{y}<extra></extra>",
            )
        )

    if df_fills is not None and {"side", "price", "elapsed_s"}.issubset(df_fills.columns):
        _add_markers(df_fills[df_fills["side"] == "buy"], "buy fills (engine)", "green", "circle")
        _add_markers(df_fills[df_fills["side"] == "sell"], "sell fills (engine)", "red", "circle")
    if df_cancels is not None and {"price", "elapsed_s"}.issubset(df_cancels.columns):
        _add_markers(df_cancels, "cancels (engine)", "gray", "x")
    if df_rejects is not None and {"price", "elapsed_s"}.issubset(df_rejects.columns):
        if "side" in df_rejects.columns:
            _add_markers(
                df_rejects[df_rejects["side"] == "buy"],
                "buy rejects (engine)",
                "orange",
                "x",
            )
            _add_markers(
                df_rejects[df_rejects["side"] == "sell"],
                "sell rejects (engine)",
                "purple",
                "x",
            )
        else:
            _add_markers(df_rejects, "rejects (engine)", "orange", "x")

    fig.update_layout(
        title="Price deltas vs strike",
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


def _build_engine_order_markers(
    exec_records: list[dict],
    df_strategy: pd.DataFrame,
    side: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "decision_id" not in df_strategy.columns:
        return (
            pd.DataFrame(columns=["side", "price", "elapsed_s", "__line__", "__lines__"]),
            pd.DataFrame(columns=["price", "elapsed_s", "__line__", "__lines__"]),
            pd.DataFrame(columns=["side", "price", "elapsed_s", "__line__", "__lines__"]),
        )

    k = (
        pd.to_numeric(df_strategy["price_to_beat"], errors="coerce")
        if "price_to_beat" in df_strategy.columns
        else pd.Series([np.nan] * len(df_strategy))
    )
    rtds_delta = (
        pd.to_numeric(df_strategy["btc_price"], errors="coerce") - k
        if "btc_price" in df_strategy.columns
        else pd.Series([np.nan] * len(df_strategy))
    )
    binance_delta = (
        pd.to_numeric(df_strategy["binance_price"], errors="coerce") - k
        if "binance_price" in df_strategy.columns
        else pd.Series([np.nan] * len(df_strategy))
    )

    decision_df = df_strategy[["decision_id", "elapsed_s"]].copy()
    decision_df["rtds_delta"] = rtds_delta.values
    decision_df["binance_delta"] = binance_delta.values
    decision_df = decision_df.dropna(subset=["decision_id", "elapsed_s"])
    decision_map = {
        int(row["decision_id"]): row for _, row in decision_df.iterrows()
    }

    def _to_int(val) -> int | None:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _norm_outcome(val) -> str | None:
        if val is None:
            return None
        txt = str(val).strip().lower()
        if txt in {"up", "yes", "y", "true", "1"}:
            return "up"
        if txt in {"down", "no", "n", "false", "0"}:
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

    side_norm = side.lower()
    orders: dict[tuple, dict] = {}

    for rec in exec_records:
        client_id = _to_int(rec.get("client_order_id"))
        strat_id = _to_int(rec.get("strategy_order_id"))
        market_id = rec.get("market_id")
        order_id = client_id if client_id is not None else strat_id
        if order_id is None:
            continue
        key = (order_id, market_id)

        side_val = _norm_side(rec.get("side"))
        outcome_val = _norm_outcome(rec.get("outcome"))
        status = str(rec.get("status") or "").strip().lower()
        decision_id = _to_int(rec.get("decision_id"))

        order = orders.setdefault(
            key,
            {
                "side": None,
                "outcome": None,
                "decision_id": None,
                "status_final": None,
                "lines": set(),
            },
        )
        line_no = rec.get("__line__")
        if line_no is not None:
            try:
                order["lines"].add(int(line_no))
            except (TypeError, ValueError):
                pass

        if side_val:
            order["side"] = side_val
        if outcome_val:
            order["outcome"] = outcome_val
        if decision_id is not None:
            order["decision_id"] = decision_id

        if status in {"filled", "partiallyfilled"}:
            order["status_final"] = "filled"
        elif status == "canceled":
            order["status_final"] = "canceled"
        elif status in {"rejected", "failed"}:
            order["status_final"] = "rejected"

    fill_rows: list[dict] = []
    cancel_rows: list[dict] = []
    reject_rows: list[dict] = []

    for order in orders.values():
        if order.get("outcome") != side_norm:
            continue
        decision_id = order.get("decision_id")
        if decision_id is None or decision_id not in decision_map:
            continue
        row = decision_map[decision_id]
        elapsed = row["elapsed_s"]
        y_val = row["rtds_delta"]
        if pd.isna(y_val):
            y_val = row["binance_delta"]
        if pd.isna(y_val):
            continue

        lines = order.get("lines") or set()
        lines_str = ",".join(str(v) for v in sorted(lines)) if lines else None
        entry = {
            "side": order.get("side"),
            "price": y_val,
            "elapsed_s": elapsed,
            "__lines__": lines_str,
        }
        status_final = order.get("status_final")
        if status_final == "filled":
            fill_rows.append(entry)
        elif status_final == "canceled":
            cancel_rows.append(entry)
        elif status_final == "rejected":
            reject_rows.append(entry)

    fills_df = (
        pd.DataFrame(fill_rows)
        if fill_rows
        else pd.DataFrame(columns=["side", "price", "elapsed_s", "__lines__"])
    )
    cancels_df = (
        pd.DataFrame(cancel_rows)
        if cancel_rows
        else pd.DataFrame(columns=["price", "elapsed_s", "__lines__"])
    )
    rejects_df = (
        pd.DataFrame(reject_rows)
        if reject_rows
        else pd.DataFrame(columns=["side", "price", "elapsed_s", "__lines__"])
    )
    return fills_df, cancels_df, rejects_df


def _plot_signal_deltas(
    df_strategy: pd.DataFrame,
    *,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    cols = [
        c
        for c in (
            "binance_w500_delta",
            "signal_delta",
            "binance_w100.delta",
            "binance_w500.delta_from_max",
            "binance_w500.delta_from_min",
        )
        if c in df_strategy.columns
    ]
    if not cols:
        return None
    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )
    for col in cols:
        fig.add_trace(
            go.Scatter(
                x=df_strategy[elapsed_col],
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
        title="Signal / window deltas",
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


def _plot_binance_w1000(
    df_strategy: pd.DataFrame,
    *,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    cols = [
        c
        for c in (
            "binance_w1000.delta_from_max",
            "binance_w1000.delta_from_min",
        )
        if c in df_strategy.columns
    ]
    if not cols:
        return None
    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )
    for col in cols:
        fig.add_trace(
            go.Scatter(
                x=df_strategy[elapsed_col],
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
        title="Binance 1000ms window stats",
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


def _plot_orderbook_w5000(
    df_strategy: pd.DataFrame,
    *,
    side: str,
    width: int,
    height: int,
    x_tick: int,
    elapsed_col: str,
) -> go.Figure | None:
    if elapsed_col not in df_strategy.columns:
        return None
    side_key = side.lower()
    columns = []
    for prefix in (f"{side_key}_bid_w5000", f"{side_key}_ask_w5000"):
        for suffix in ("delta_from_max", "delta_from_min"):
            col = f"{prefix}.{suffix}"
            if col in df_strategy.columns:
                columns.append(col)

    if not columns:
        return None

    fig = go.Figure()
    line_meta = (
        df_strategy.get("__line__", pd.Series([np.nan] * len(df_strategy)))
        .astype("Int64")
        .tolist()
    )
    for col in columns:
        fig.add_trace(
            go.Scatter(
                x=df_strategy[elapsed_col],
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
        title=f"{side} orderbook 5000ms window",
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


def _parse_execution_new(
    exec_records: list[dict],
    side: str,
    base_time: pd.Timestamp,
    *,
    end_time_s: float | None = None,
    ts_field: str = "execution_ts_ms",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        if txt in {"up", "yes", "y", "true", "1"}:
            return "up"
        if txt in {"down", "no", "n", "false", "0"}:
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

    def _elapsed_from_ts(ts_val: int | None) -> float | None:
        if ts_val is None:
            return None
        dt = _safe_ms_to_datetime(pd.Series([ts_val])).iloc[0]
        if pd.isna(dt):
            return None
        return (dt - base_time).total_seconds()

    orders: dict[tuple, dict] = {}

    for rec in exec_records:
        client_id = _to_int(rec.get("client_order_id"))
        strat_id = _to_int(rec.get("strategy_order_id"))
        market_id = rec.get("market_id")
        order_id = client_id if client_id is not None else strat_id
        if order_id is None:
            continue
        key = (order_id, market_id)

        ts_val = rec.get(ts_field)
        side_val = _norm_side(rec.get("side"))
        outcome_val = _norm_outcome(rec.get("outcome"))
        order_type = rec.get("order_type")
        order_type = str(order_type).strip().lower() if order_type is not None else None
        status = str(rec.get("status") or "").strip().lower()
        limit_price = rec.get("limit_price")
        avg_price = rec.get("avg_price")
        filled_amount = rec.get("filled_amount")
        decision_id = rec.get("decision_id")

        order = orders.setdefault(
            key,
            {
                "created_ts": None,
                "last_ts": None,
                "side": None,
                "outcome": None,
                "limit_price": None,
                "order_type": None,
                "decision_id": None,
                "status_final": None,
                "filled_amount": None,
                "avg_price": None,
                "filled_ts": None,
                "canceled_ts": None,
                "accepted_ts": None,
                "rejected_ts": None,
                "created_line": None,
                "filled_line": None,
                "canceled_line": None,
                "accepted_line": None,
                "rejected_line": None,
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
            if order["created_ts"] is None:
                order["created_ts"] = ts_val
                order["created_line"] = line_no

        if side_val:
            order["side"] = side_val
        if outcome_val:
            order["outcome"] = outcome_val
        if limit_price is not None:
            order["limit_price"] = limit_price
        if order_type:
            order["order_type"] = order_type
        if decision_id is not None:
            order["decision_id"] = decision_id
        if filled_amount is not None:
            order["filled_amount"] = filled_amount
        if avg_price is not None:
            order["avg_price"] = avg_price

        if status in {"accepted", "placed"}:
            if ts_val is not None and order["accepted_ts"] is None:
                order["accepted_ts"] = ts_val
                order["accepted_line"] = line_no
        elif status in {"rejected", "failed"}:
            if ts_val is not None:
                order["rejected_ts"] = ts_val
                order["rejected_line"] = line_no
                order["status_final"] = "rejected"
        elif status in {"filled", "partiallyfilled"}:
            order["status_final"] = "filled"
            if ts_val is not None:
                order["filled_ts"] = ts_val
                order["filled_line"] = line_no
        elif status == "canceled":
            order["status_final"] = "canceled"
            if ts_val is not None:
                order["canceled_ts"] = ts_val
                order["canceled_line"] = line_no

    fill_rows: list[dict] = []
    cancel_rows: list[dict] = []
    limit_rows: list[dict] = []
    accept_rows: list[dict] = []
    reject_rows: list[dict] = []

    for order in orders.values():
        outcome_val = order.get("outcome")
        if outcome_val is None or outcome_val != side_norm:
            continue

        side_val = order.get("side")
        limit_price = order.get("limit_price")
        avg_price = order.get("avg_price")
        order_type = order.get("order_type")
        status_final = order.get("status_final") or "open"
        lines = order.get("lines") or set()
        lines_str = ",".join(str(v) for v in sorted(lines)) if lines else None

        created_ts = order.get("created_ts") or order.get("last_ts")
        filled_ts = order.get("filled_ts")
        canceled_ts = order.get("canceled_ts")
        accepted_ts = order.get("accepted_ts")
        rejected_ts = order.get("rejected_ts")

        if accepted_ts is not None:
            elapsed = _elapsed_from_ts(accepted_ts)
            if elapsed is not None and limit_price is not None:
                accept_rows.append(
                    {
                        "side": side_val,
                        "price": limit_price,
                        "elapsed_s": elapsed,
                        "__line__": order.get("accepted_line"),
                        "__lines__": lines_str,
                    }
                )

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

        if rejected_ts is not None and limit_price is not None:
            elapsed = _elapsed_from_ts(rejected_ts)
            if elapsed is not None:
                reject_rows.append(
                    {
                        "side": side_val,
                        "price": limit_price,
                        "elapsed_s": elapsed,
                        "__line__": order.get("rejected_line"),
                        "__lines__": lines_str,
                    }
                )

        if order_type in {"gtc", "gtd"} and created_ts is not None and limit_price is not None:
            start_s = _elapsed_from_ts(created_ts)
            end_ts = filled_ts or canceled_ts
            end_s = _elapsed_from_ts(end_ts) if end_ts is not None else end_time_s
            if start_s is not None and end_s is not None:
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
    accepts_df = (
        pd.DataFrame(accept_rows)
        if accept_rows
        else pd.DataFrame(columns=["side", "price", "elapsed_s", "__line__", "__lines__"])
    )
    rejects_df = (
        pd.DataFrame(reject_rows)
        if reject_rows
        else pd.DataFrame(columns=["side", "price", "elapsed_s", "__line__", "__lines__"])
    )
    return fills_df, cancels_df, limits_df, accepts_df, rejects_df


st.set_page_config(page_title="Market Combo Dashboard (New Logs)", layout="wide")

st.title("Market Combo Dashboard (New Logs: Strategy + Execution)")

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
        placeholder="/path/to/output/json/markets/market_<window_start_sec>",
    )
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

if not market_dir:
    st.info("Set market directory in the sidebar.")
    st.stop()

if not sides:
    st.info("Select at least one side (Up/Down) in the sidebar.")
    st.stop()

market_path = Path(market_dir)
strategy_path = market_path / "strategy.jsonl"
execution_path = market_path / "execution.jsonl"

if not strategy_path.exists():
    st.error(f"Missing strategy.jsonl: {strategy_path}")
    st.stop()

has_execution = execution_path.exists()
if not has_execution:
    st.warning(
        f"execution.jsonl not found: {execution_path}. "
        "Rendering charts without order markers."
    )

with st.spinner("Loading strategy..."):
    strategy_records = load_jsonl(strategy_path)
    df_strategy = pd.json_normalize(strategy_records, sep=".")
    if df_strategy.empty:
        st.error("No strategy records found.")
        st.stop()

    ts_ms = _strategy_timestamp(df_strategy)
    if ts_ms.empty or ts_ms.notna().sum() == 0:
        st.error("No valid timestamps found in strategy.jsonl")
        st.stop()

    df_strategy["timestamp"] = _safe_ms_to_datetime(ts_ms)
    if "best_up_bid" in df_strategy.columns and "up_bid" not in df_strategy.columns:
        df_strategy["up_bid"] = df_strategy["best_up_bid"]
    if "best_up_ask" in df_strategy.columns and "up_ask" not in df_strategy.columns:
        df_strategy["up_ask"] = df_strategy["best_up_ask"]
    if "best_down_bid" in df_strategy.columns and "down_bid" not in df_strategy.columns:
        df_strategy["down_bid"] = df_strategy["best_down_bid"]
    if "best_down_ask" in df_strategy.columns and "down_ask" not in df_strategy.columns:
        df_strategy["down_ask"] = df_strategy["best_down_ask"]

    df_strategy = maybe_downsample(df_strategy, rule=downsample, max_points=int(max_points))

exec_records: list[dict] = []
if has_execution:
    with st.spinner("Loading execution..."):
        try:
            exec_records = load_jsonl(execution_path)
        except (FileNotFoundError, ValueError) as exc:
            st.warning(
                f"Failed to load execution.jsonl ({exc}). "
                "Rendering charts without order markers."
            )
            exec_records = []

exec_ts_series = pd.to_numeric(
    pd.Series([rec.get("execution_ts_ms") for rec in exec_records if rec.get("execution_ts_ms") is not None]),
    errors="coerce",
)

strategy_min = df_strategy["timestamp"].min()
exec_min = (
    _safe_ms_to_datetime(pd.Series([exec_ts_series.min()])).iloc[0]
    if exec_ts_series.notna().any()
    else None
)
base_candidates = [ts for ts in [strategy_min, exec_min] if ts is not None and pd.notna(ts)]
if base_candidates:
    base_time = min(base_candidates)
else:
    st.error("No valid timestamps found in strategy or execution.")
    st.stop()

df_strategy["elapsed_s"] = (df_strategy["timestamp"] - base_time).dt.total_seconds()
end_time_s = float(df_strategy["elapsed_s"].max())

with st.spinner("Parsing execution..."):
    df_exec_up, df_cancels_up, df_limits_up, df_accepts_up, df_rejects_up = _parse_execution_new(
        exec_records, side="Up", base_time=base_time, end_time_s=end_time_s
    )
    df_exec_down, df_cancels_down, df_limits_down, df_accepts_down, df_rejects_down = _parse_execution_new(
        exec_records, side="Down", base_time=base_time, end_time_s=end_time_s
    )
    df_engine_up, df_engine_cancels_up, df_engine_rejects_up = _build_engine_order_markers(
        exec_records, df_strategy, side="Up"
    )
    df_engine_down, df_engine_cancels_down, df_engine_rejects_down = _build_engine_order_markers(
        exec_records, df_strategy, side="Down"
    )
    df_engine_fills = pd.concat([df_engine_up, df_engine_down], ignore_index=True)
    df_engine_cancels = pd.concat([df_engine_cancels_up, df_engine_cancels_down], ignore_index=True)
    df_engine_rejects = pd.concat([df_engine_rejects_up, df_engine_rejects_down], ignore_index=True)

width = int(base_width * xscale)

for s in sides:
    df_exec_side = df_exec_up if s.lower() == "up" else df_exec_down
    df_cancels_side = df_cancels_up if s.lower() == "up" else df_cancels_down
    df_limits_side = df_limits_up if s.lower() == "up" else df_limits_down
    df_accepts_side = df_accepts_up if s.lower() == "up" else df_accepts_down
    df_rejects_side = df_rejects_up if s.lower() == "up" else df_rejects_down
    fig = _plot_book_with_events(
        df_strategy,
        df_exec_side,
        df_cancels_side,
        df_limits_side,
        df_accepts_side,
        df_rejects_side,
        side=s,
        width=width,
        height=int(height),
        elapsed_col="elapsed_s",
    )
    fig.update_xaxes(dtick=int(x_tick))
    st.plotly_chart(fig, width=width, config={"responsive": False})

    w5000_fig = _plot_orderbook_w5000(
        df_strategy,
        side=s,
        width=width,
        height=int(height),
        x_tick=int(x_tick),
        elapsed_col="elapsed_s",
    )
    if w5000_fig is not None:
        st.plotly_chart(w5000_fig, width=width, config={"responsive": False})

price_fig = _plot_price_vs_k(
    df_strategy,
    width=width,
    height=int(height),
    x_tick=int(x_tick),
    elapsed_col="elapsed_s",
)
if price_fig is not None:
    st.plotly_chart(price_fig, width=width, config={"responsive": False})

delta_fig = _plot_price_deltas(
    df_strategy,
    width=width,
    height=int(height),
    x_tick=int(x_tick),
    elapsed_col="elapsed_s",
    df_fills=df_engine_fills,
    df_cancels=df_engine_cancels,
    df_rejects=df_engine_rejects,
)
if delta_fig is not None:
    st.plotly_chart(delta_fig, width=width, config={"responsive": False})

signal_fig = _plot_signal_deltas(
    df_strategy,
    width=width,
    height=int(height),
    x_tick=int(x_tick),
    elapsed_col="elapsed_s",
)
if signal_fig is not None:
    st.plotly_chart(signal_fig, width=width, config={"responsive": False})

win1000_fig = _plot_binance_w1000(
    df_strategy,
    width=width,
    height=int(height),
    x_tick=int(x_tick),
    elapsed_col="elapsed_s",
)
if win1000_fig is not None:
    st.plotly_chart(win1000_fig, width=width, config={"responsive": False})
