from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlotSpec:
    name: str
    columns: tuple[str, ...]
    title: str
    ylabel: str | None = None
    price_axis_01: bool = False
    time_source: str = "default"


MIN_VALID_MS = 0  # Unix epoch and later
MAX_VALID_MS = 9_223_372_036_854  # pandas max Timestamp in ms (2262-04-11)


NUMERIC_PLOTS: tuple[PlotSpec, ...] = (
    PlotSpec(
        name="edges_thresholds",
        columns=(
            "edge_buy_up",
            "edge_buy_down",
            "edge_hold_up",
            "edge_hold_down",
            "thr_enter",
            "thr_exit",
            "thr_taker",
        ),
        title="Edge vs thresholds",
        ylabel="edge / threshold",
    ),
    PlotSpec(
        name="fair_vs_book",
        columns=(
            "fair_p_up",
            "fair_p_down",
            "up_bid",
            "up_ask",
            "down_bid",
            "down_ask",
            "price",
        ),
        title="Fair probabilities and book",
        price_axis_01=True,
    ),
    PlotSpec(
        name="risk_sigma",
        columns=(
            "sigma_selected",
            "sigma_short",
            "sigma_long",
            "shock_ratio",
            "move_5s",
        ),
        title="Risk / volatility",
    ),
    PlotSpec(
        name="positions",
        columns=(
            "position.up_tokens",
            "position.down_tokens",
            "position.usdc_balance",
            "exposure_size",
        ),
        title="Position / balance",
    ),
)

TELEMETRY_PLOTS: tuple[PlotSpec, ...] = (
    PlotSpec(
        name="telemetry_btc_price",
        columns=(
            "btc.price",
            "btc.price_to_beat",
        ),
        title="BTC price vs strike",
        ylabel="price",
        time_source="btc",
    ),
    PlotSpec(
        name="telemetry_fair_vs_mid",
        columns=(
            "btc.p_fair_up",
            "btc.p_fair_down",
            "up_l1.mid",
            "down_l1.mid",
        ),
        title="Fair probability vs L1 mid",
        ylabel="probability",
        price_axis_01=True,
        time_source="event",
    ),
    PlotSpec(
        name="telemetry_shock",
        columns=(
            "shock.shock_ratio",
            "shock.move_5s",
            "btc.z_impulse_r1",
            "btc.z_delta",
        ),
        title="Shock / impulse metrics",
        time_source="btc",
    ),
)

TELEMETRY_BEST_BOOK_PLOTS: tuple[PlotSpec, ...] = (
    PlotSpec(
        name="telemetry_up_l1_bid_ask",
        columns=("up_l1.bid_p", "up_l1.ask_p", "up_l1.mid"),
        title="Up L1 bid/ask",
        ylabel="price",
        price_axis_01=True,
        time_source="up_book",
    ),
    PlotSpec(
        name="telemetry_down_l1_bid_ask",
        columns=("down_l1.bid_p", "down_l1.ask_p", "down_l1.mid"),
        title="Down L1 bid/ask",
        ylabel="price",
        price_axis_01=True,
        time_source="down_book",
    ),
    PlotSpec(
        name="telemetry_btc_delta",
        columns=("btc.delta", "btc.delta_pct"),
        title="BTC delta (price - strike)",
        time_source="btc",
    ),
)

SPIKE_LAG_PLOTS: tuple[PlotSpec, ...] = (
    PlotSpec(
        name="spike_btc_price",
        columns=("btc_price", "price_to_beat", "coinbase_price"),
        title="BTC price vs strike",
        ylabel="price",
        time_source="btc",
    ),
    PlotSpec(
        name="spike_btc_delta",
        columns=("btc_delta", "coinbase_delta"),
        title="BTC delta vs strike",
        ylabel="delta",
        time_source="btc",
    ),
    PlotSpec(
        name="spike_fair_lag",
        columns=("p_fair_up", "dp_fair_up", "lag_react_up", "lag_react_down", "cross_strength"),
        title="Fair / lag metrics",
        time_source="event",
    ),
    PlotSpec(
        name="spike_book_up",
        columns=("up_bid", "up_ask", "micro_up"),
        title="Up book (best bid/ask)",
        ylabel="price",
        price_axis_01=True,
        time_source="up_book",
    ),
    PlotSpec(
        name="spike_book_down",
        columns=("down_bid", "down_ask", "micro_down"),
        title="Down book (best bid/ask)",
        ylabel="price",
        price_axis_01=True,
        time_source="down_book",
    ),
    PlotSpec(
        name="spike_gating",
        columns=(
            "tau_s",
            "tau_min_s",
            "tau_max_s",
            "tau_flat_s",
            "btc_tick_age_ms",
            "max_btc_tick_age_ms",
        ),
        title="Entry gating / timing",
        time_source="event",
    ),
    PlotSpec(
        name="spike_thresholds",
        columns=("entry_lag_threshold", "taker_lag_threshold"),
        title="Entry thresholds",
        time_source="event",
    ),
    PlotSpec(
        name="spike_spreads",
        columns=("spread_up_ticks", "spread_down_ticks"),
        title="Spread (ticks)",
        time_source="book",
    ),
)

CATEGORY_PLOTS: tuple[tuple[str, str], ...] = (
    ("decision", "Decision timeline"),
    ("exposure", "Exposure timeline"),
)


BINARY_PLOTS: tuple[tuple[str, str], ...] = (
    ("shock_mode", "Shock mode"),
)


def _ensure_timestamp(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns:
        raise ValueError("Missing 'timestamp' column for plotting")


def _elapsed_seconds(
    timestamps: pd.Series,
    base_time: pd.Timestamp | None = None,
) -> pd.Series:
    if timestamps.empty:
        return pd.Series(dtype=float)

    ts = pd.to_datetime(timestamps, errors="coerce")
    if ts.empty:
        return pd.Series(dtype=float)

    if not ts.notna().any():
        return pd.Series([float("nan")] * len(ts), index=ts.index)

    if hasattr(ts.dtype, "tz") and ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)

    if base_time is not None and getattr(base_time, "tzinfo", None) is not None:
        base_time = base_time.tz_convert(None)

    start = base_time if base_time is not None else ts.min()
    if pd.isna(start):
        return pd.Series([float("nan")] * len(ts), index=ts.index)

    return (ts - start).dt.total_seconds()


def _base_time(df: pd.DataFrame) -> pd.Timestamp | None:
    candidates: list[pd.Series] = []
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            candidates.append(ts)

    ms_columns = (
        "ts_ms",
        "event_ts_ms.log",
        "event_ts_ms.btc",
        "event_ts_ms.book_up",
        "event_ts_ms.book_down",
        "event_ts_ms.book",
        "btc.ts_ms",
        "btc_ts_ms",
        "telemetry_state.btc.last_sample.ts_ms",
        "up_l1.ts_ms",
        "down_l1.ts_ms",
        "telemetry_state.book.up.last_sample.ts_ms",
        "telemetry_state.book.down.last_sample.ts_ms",
    )
    for col in ms_columns:
        if col in df.columns:
            series = _safe_ms(df[col])
            if series.notna().any():
                candidates.append(_ms_to_datetime(series))

    if not candidates:
        return None

    combined = pd.concat(candidates, axis=1)
    base = combined.min().min()
    if pd.isna(base):
        return None
    if getattr(base, "tzinfo", None) is not None:
        base = base.tz_convert(None)
    return base


def _ms_series(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series | None:
    for col in columns:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                return series
    return None


def _safe_ms(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.where(series.between(MIN_VALID_MS, MAX_VALID_MS))
    return series


def _ms_to_datetime(series: pd.Series) -> pd.Series:
    safe = _safe_ms(series)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return pd.to_datetime(
                safe.astype("float64"), unit="ms", utc=True, errors="coerce"
            )
    except FloatingPointError:
        return pd.to_datetime(
            pd.Series([np.nan] * len(safe), index=safe.index),
            unit="ms",
            utc=True,
            errors="coerce",
        )


def _select_time_series(df: pd.DataFrame, source: str) -> pd.Series:
    if source == "default":
        if "timestamp" in df.columns:
            return df["timestamp"]
        return pd.Series([pd.NaT] * len(df), index=df.index)

    if source == "btc":
        btc_ms = _ms_series(
            df,
            (
                "event_ts_ms.btc",
                "btc.ts_ms",
                "btc_ts_ms",
                "telemetry_state.btc.last_sample.ts_ms",
            ),
        )
        if btc_ms is not None:
            return _ms_to_datetime(btc_ms)
        return _select_time_series(df, "default")

    if source == "up_book":
        up_ms = _ms_series(
            df,
            (
                "event_ts_ms.book_up",
                "up_l1.ts_ms",
                "telemetry_state.book.up.last_sample.ts_ms",
            ),
        )
        if up_ms is not None:
            return _ms_to_datetime(up_ms)
        return _select_time_series(df, "default")

    if source == "down_book":
        down_ms = _ms_series(
            df,
            (
                "event_ts_ms.book_down",
                "down_l1.ts_ms",
                "telemetry_state.book.down.last_sample.ts_ms",
            ),
        )
        if down_ms is not None:
            return _ms_to_datetime(down_ms)
        return _select_time_series(df, "default")

    if source in {"book", "event"}:
        book_ms_direct = _ms_series(df, ("event_ts_ms.book",))
        up_ms = _ms_series(
            df,
            (
                "event_ts_ms.book_up",
                "up_l1.ts_ms",
                "telemetry_state.book.up.last_sample.ts_ms",
            ),
        )
        down_ms = _ms_series(
            df,
            (
                "event_ts_ms.book_down",
                "down_l1.ts_ms",
                "telemetry_state.book.down.last_sample.ts_ms",
            ),
        )
        btc_ms = _ms_series(
            df,
            (
                "event_ts_ms.btc",
                "btc.ts_ms",
                "btc_ts_ms",
                "telemetry_state.btc.last_sample.ts_ms",
            ),
        )

        frames = []
        if book_ms_direct is not None:
            book_ms = book_ms_direct
        else:
            if up_ms is not None:
                frames.append(up_ms)
            if down_ms is not None:
                frames.append(down_ms)
            book_ms = None
            if frames:
                book_ms = pd.concat(frames, axis=1).max(axis=1, skipna=True)

        if source == "book":
            if book_ms is not None and book_ms.notna().any():
                return _ms_to_datetime(book_ms)
            if btc_ms is not None:
                return _ms_to_datetime(btc_ms)
            return _select_time_series(df, "default")

        frames = []
        if book_ms is not None:
            frames.append(book_ms)
        if btc_ms is not None:
            frames.append(btc_ms)
        if frames:
            event_ms = pd.concat(frames, axis=1).max(axis=1, skipna=True)
            return _ms_to_datetime(event_ms)

        return _select_time_series(df, "default")

    return _select_time_series(df, "default")

def _format_elapsed_axis(ax: plt.Axes, elapsed_seconds: pd.Series) -> None:
    if elapsed_seconds.empty:
        return

    max_seconds = elapsed_seconds.max()
    if pd.isna(max_seconds):
        return

    total_seconds = max(float(max_seconds), 1.0)

    interval = 5 if total_seconds >= 5 else 1
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=interval))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{int(x)}")
    )
    ax.tick_params(axis="x", rotation=30)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(base=1.0))
    ax.grid(True, axis="x", which="minor", alpha=0.1)
    ax.grid(True, axis="x", which="major", alpha=0.2)
    ax.set_xlabel("seconds from start")


def _format_price_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: f"{y:.2f}")
    )
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.grid(True, axis="y", which="minor", alpha=0.15)


def _save_figure(
    fig: plt.Figure,
    out_path: Path,
    formats: Iterable[str],
    *,
    dpi: int,
) -> list[Path]:
    out_paths: list[Path] = []
    for fmt in formats:
        target = out_path.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        out_paths.append(target)
    return out_paths


def _plot_numeric(
    df: pd.DataFrame,
    spec: PlotSpec,
    out_dir: Path,
    formats: Iterable[str],
    figscale: float,
    xscale: float,
    yscale: float,
    dpi: int,
    base_time: pd.Timestamp | None,
) -> list[Path]:
    available = [col for col in spec.columns if col in df.columns]
    if not available:
        return []

    time_series = _select_time_series(df, spec.time_source)
    x = _elapsed_seconds(time_series, base_time)
    fig, ax = plt.subplots(
        figsize=(12 * figscale * xscale, 4 * figscale * yscale)
    )
    for col in available:
        series = pd.to_numeric(df[col], errors="coerce")
        ax.plot(x, series, label=col, linewidth=1)

    ax.set_title(spec.title)
    if spec.ylabel:
        ax.set_ylabel(spec.ylabel)
    ax.grid(True, alpha=0.3)
    _format_elapsed_axis(ax, x)
    if spec.price_axis_01:
        _format_price_axis(ax)
    if len(available) > 1:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
    fig.tight_layout()

    out_path = out_dir / spec.name
    paths = _save_figure(fig, out_path, formats, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_categorical(
    df: pd.DataFrame,
    column: str,
    title: str,
    out_dir: Path,
    formats: Iterable[str],
    figscale: float,
    xscale: float,
    yscale: float,
    dpi: int,
    base_time: pd.Timestamp | None,
) -> list[Path]:
    if column not in df.columns:
        return []

    categories = df[column].astype("category")
    codes = categories.cat.codes

    fig, ax = plt.subplots(
        figsize=(12 * figscale * xscale, 3 * figscale * yscale)
    )
    time_series = _select_time_series(df, "default")
    x = _elapsed_seconds(time_series, base_time)
    ax.scatter(x, codes, s=8)
    ax.set_title(title)
    ax.set_yticks(range(len(categories.cat.categories)))
    ax.set_yticklabels([str(cat) for cat in categories.cat.categories])
    ax.grid(True, axis="y", alpha=0.3)
    _format_elapsed_axis(ax, x)
    fig.tight_layout()

    out_path = out_dir / column
    paths = _save_figure(fig, out_path, formats, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_binary(
    df: pd.DataFrame,
    column: str,
    title: str,
    out_dir: Path,
    formats: Iterable[str],
    figscale: float,
    xscale: float,
    yscale: float,
    dpi: int,
    base_time: pd.Timestamp | None,
) -> list[Path]:
    if column not in df.columns:
        return []

    series = pd.to_numeric(df[column], errors="coerce").fillna(0)

    fig, ax = plt.subplots(
        figsize=(12 * figscale * xscale, 2.5 * figscale * yscale)
    )
    time_series = _select_time_series(df, "default")
    x = _elapsed_seconds(time_series, base_time)
    ax.step(x, series, where="post")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    _format_elapsed_axis(ax, x)
    fig.tight_layout()

    out_path = out_dir / column
    paths = _save_figure(fig, out_path, formats, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_orders_vs_book(
    df: pd.DataFrame,
    out_dir: Path,
    formats: Iterable[str],
    figscale: float,
    xscale: float,
    yscale: float,
    dpi: int,
    base_time: pd.Timestamp | None,
) -> list[Path]:
    if "orders" not in df.columns:
        return []

    book_columns = [
        col
        for col in ("up_bid", "up_ask", "down_bid", "down_ask", "price")
        if col in df.columns
    ]
    if not book_columns:
        return []

    fig, ax = plt.subplots(
        figsize=(12 * figscale * xscale, 4 * figscale * yscale)
    )

    time_series = _select_time_series(df, "book")
    x = _elapsed_seconds(time_series, base_time)
    for col in book_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        ax.plot(x, series, label=col, linewidth=1)

    order_points = {"Up": [], "Down": [], "Other": []}
    order_meta = {"Up": [], "Down": [], "Other": []}

    for elapsed_ts, orders in zip(x, df["orders"]):
        if not isinstance(orders, list):
            continue
        for order in orders:
            if not isinstance(order, dict):
                continue
            price = order.get("limit_price", order.get("price"))
            if price is None:
                continue
            outcome = str(order.get("outcome", "Other"))
            side_raw = str(order.get("side", ""))
            side_lower = side_raw.strip().lower()
            if side_lower == "buy":
                side = "Buy"
            elif side_lower == "sell":
                side = "Sell"
            else:
                side = "Other"
            key = "Other"
            if outcome.lower() == "up":
                key = "Up"
            elif outcome.lower() == "down":
                key = "Down"
            order_points[key].append((elapsed_ts, price))
            order_meta[key].append(side)

    color_map = {"Buy": "tab:green", "Sell": "tab:red", "Other": "tab:gray"}
    marker_map = {"Up": "^", "Down": "v", "Other": "o"}

    for key, points in order_points.items():
        if not points:
            continue
        xs, ys = zip(*points)
        sides = order_meta[key]
        for side_value in set(sides):
            mask = [side == side_value for side in sides]
            marker = marker_map.get(key, "o")
            label = f"orders {key} {side_value}".strip()
            ax.scatter(
                [x for x, keep in zip(xs, mask) if keep],
                [y for y, keep in zip(ys, mask) if keep],
                s=20,
                marker=marker,
                color=color_map.get(side_value, "tab:gray"),
                edgecolors="black",
                linewidths=0.3,
                label=label,
                alpha=0.8,
            )

    ax.set_title("Orders vs order book")
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.3)
    _format_elapsed_axis(ax, x)
    _format_price_axis(ax)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()

    out_path = out_dir / "orders_vs_book"
    paths = _save_figure(fig, out_path, formats, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_trade_bands(
    df: pd.DataFrame,
    out_dir: Path,
    formats: Iterable[str],
    figscale: float,
    xscale: float,
    yscale: float,
    dpi: int,
    base_time: pd.Timestamp | None,
) -> list[Path]:
    if "thr_enter" not in df.columns or "thr_exit" not in df.columns:
        return []

    created: list[Path] = []

    if "fair_p_up" in df.columns and ("up_bid" in df.columns or "up_ask" in df.columns):
        fair_up = pd.to_numeric(df["fair_p_up"], errors="coerce")
        thr_enter = pd.to_numeric(df["thr_enter"], errors="coerce")
        thr_exit = pd.to_numeric(df["thr_exit"], errors="coerce")
        buy_level = fair_up - thr_enter
        sell_level = fair_up - thr_exit

        fig, ax = plt.subplots(
            figsize=(12 * figscale * xscale, 4 * figscale * yscale)
        )
        time_series = _select_time_series(df, "event")
        x = _elapsed_seconds(time_series, base_time)
        if "up_bid" in df.columns:
            ax.plot(x, pd.to_numeric(df["up_bid"], errors="coerce"), label="up_bid")
        if "up_ask" in df.columns:
            ax.plot(x, pd.to_numeric(df["up_ask"], errors="coerce"), label="up_ask")
        ax.plot(x, buy_level, label="buy_level_up (fair - thr_enter)")
        ax.plot(x, sell_level, label="sell_level_up (fair - thr_exit)")

        ax.set_title("Trade bands (Up)")
        ax.set_ylabel("price")
        ax.grid(True, alpha=0.3)
        _format_elapsed_axis(ax, x)
        _format_price_axis(ax)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout()

        out_path = out_dir / "trade_bands_up"
        created.extend(_save_figure(fig, out_path, formats, dpi=dpi))
        plt.close(fig)

    if "fair_p_down" in df.columns or "fair_p_up" in df.columns:
        fair_down = None
        if "fair_p_down" in df.columns:
            fair_down = pd.to_numeric(df["fair_p_down"], errors="coerce")
        elif "fair_p_up" in df.columns:
            fair_down = 1.0 - pd.to_numeric(df["fair_p_up"], errors="coerce")

        if fair_down is not None and ("down_bid" in df.columns or "down_ask" in df.columns):
            thr_enter = pd.to_numeric(df["thr_enter"], errors="coerce")
            thr_exit = pd.to_numeric(df["thr_exit"], errors="coerce")
            buy_level = fair_down - thr_enter
            sell_level = fair_down - thr_exit

            fig, ax = plt.subplots(
                figsize=(12 * figscale * xscale, 4 * figscale * yscale)
            )
            time_series = _select_time_series(df, "event")
            x = _elapsed_seconds(time_series, base_time)
            if "down_bid" in df.columns:
                ax.plot(
                    x,
                    pd.to_numeric(df["down_bid"], errors="coerce"),
                    label="down_bid",
                )
            if "down_ask" in df.columns:
                ax.plot(
                    x,
                    pd.to_numeric(df["down_ask"], errors="coerce"),
                    label="down_ask",
                )
            ax.plot(x, buy_level, label="buy_level_down (fair - thr_enter)")
            ax.plot(x, sell_level, label="sell_level_down (fair - thr_exit)")

            ax.set_title("Trade bands (Down)")
            ax.set_ylabel("price")
            ax.grid(True, alpha=0.3)
            _format_elapsed_axis(ax, x)
            _format_price_axis(ax)
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                frameon=False,
            )
            fig.tight_layout()

            out_path = out_dir / "trade_bands_down"
            created.extend(_save_figure(fig, out_path, formats, dpi=dpi))
            plt.close(fig)

    return created


def _plot_strike_diff(
    df: pd.DataFrame,
    out_dir: Path,
    formats: Iterable[str],
    figscale: float,
    xscale: float,
    yscale: float,
    dpi: int,
    base_time: pd.Timestamp | None,
) -> list[Path]:
    strike_col = None
    if "strike_k" in df.columns:
        strike_col = "strike_k"
    elif "stripe_k" in df.columns:
        strike_col = "stripe_k"

    if strike_col is None or "btc_price" not in df.columns:
        return []

    strike = pd.to_numeric(df[strike_col], errors="coerce")
    btc = pd.to_numeric(df["btc_price"], errors="coerce")
    diff = btc - strike

    fig, ax = plt.subplots(
        figsize=(12 * figscale * xscale, 3.5 * figscale * yscale)
    )
    time_series = _select_time_series(df, "btc")
    x = _elapsed_seconds(time_series, base_time)
    ax.plot(x, diff, label="btc_price - strike_k", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Strike difference (BTC price - strike)")
    ax.set_ylabel("price diff")
    ax.grid(True, alpha=0.3)
    _format_elapsed_axis(ax, x)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()

    out_path = out_dir / "strike_diff"
    paths = _save_figure(fig, out_path, formats, dpi=dpi)
    plt.close(fig)
    return paths


def generate_plots(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    formats: Iterable[str] = ("png", "svg"),
    figscale: float = 1.0,
    xscale: float = 1.0,
    yscale: float = 1.0,
    dpi: int = 150,
) -> list[Path]:
    _ensure_timestamp(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_time = _base_time(df)
    created: list[Path] = []

    for spec in NUMERIC_PLOTS:
        created.extend(
            _plot_numeric(
                df,
                spec,
                out_dir,
                formats,
                figscale,
                xscale,
                yscale,
                dpi,
                base_time,
            )
        )

    for spec in TELEMETRY_PLOTS:
        created.extend(
            _plot_numeric(
                df,
                spec,
                out_dir,
                formats,
                figscale,
                xscale,
                yscale,
                dpi,
                base_time,
            )
        )

    for spec in TELEMETRY_BEST_BOOK_PLOTS:
        created.extend(
            _plot_numeric(
                df,
                spec,
                out_dir,
                formats,
                figscale,
                xscale,
                yscale,
                dpi,
                base_time,
            )
        )

    for spec in SPIKE_LAG_PLOTS:
        created.extend(
            _plot_numeric(
                df,
                spec,
                out_dir,
                formats,
                figscale,
                xscale,
                yscale,
                dpi,
                base_time,
            )
        )

    for column, title in CATEGORY_PLOTS:
        created.extend(
            _plot_categorical(
                df,
                column,
                title,
                out_dir,
                formats,
                figscale,
                xscale,
                yscale,
                dpi,
                base_time,
            )
        )

    for column, title in BINARY_PLOTS:
        created.extend(
            _plot_binary(
                df,
                column,
                title,
                out_dir,
                formats,
                figscale,
                xscale,
                yscale,
                dpi,
                base_time,
            )
        )

    created.extend(
        _plot_orders_vs_book(
            df, out_dir, formats, figscale, xscale, yscale, dpi, base_time
        )
    )
    created.extend(
        _plot_trade_bands(
            df, out_dir, formats, figscale, xscale, yscale, dpi, base_time
        )
    )
    created.extend(
        _plot_strike_diff(
            df, out_dir, formats, figscale, xscale, yscale, dpi, base_time
        )
    )

    return created
