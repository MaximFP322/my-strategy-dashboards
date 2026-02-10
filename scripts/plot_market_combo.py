from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plotting.io import load_jsonl
from plotting.processing import _safe_ms_to_datetime


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


def _format_price_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:.2f}"))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.grid(True, axis="y", which="minor", alpha=0.15)


def _format_time_axis(ax: plt.Axes, elapsed: pd.Series) -> None:
    if elapsed.empty or elapsed.isna().all():
        return
    max_seconds = float(elapsed.max())
    interval = 5 if max_seconds >= 5 else 1
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=interval))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(base=1.0))
    ax.grid(True, axis="x", which="minor", alpha=0.1)
    ax.grid(True, axis="x", which="major", alpha=0.2)
    ax.set_xlabel("seconds")
    ax.tick_params(axis="x", rotation=30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot combined Up book + fills + cancels from market logs",
    )
    parser.add_argument(
        "--market-dir",
        required=True,
        help="Path to market directory with strategy.jsonl and execution.jsonl",
    )
    parser.add_argument(
        "--strategy-name",
        default=None,
        help="Filter strategy.jsonl by strategy name (optional)",
    )
    parser.add_argument(
        "--out",
        default="output/plots/combined",
        help="Output directory",
    )
    parser.add_argument(
        "--formats",
        default="png,svg",
        help="Comma-separated output formats",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    market_dir = Path(args.market_dir)
    strategy_path = market_dir / "strategy.jsonl"
    execution_path = market_dir / "execution.jsonl"

    if not strategy_path.exists():
        raise FileNotFoundError(f"Missing strategy.jsonl: {strategy_path}")
    if not execution_path.exists():
        raise FileNotFoundError(f"Missing execution.jsonl: {execution_path}")

    records = load_jsonl(strategy_path)
    if args.strategy_name:
        records = [r for r in records if r.get("strategy") == args.strategy_name]

    df = pd.json_normalize(records, sep=".")
    if df.empty:
        raise ValueError("strategy.jsonl has no records after filtering")

    book_ts_ms = _book_timestamp(df)
    if book_ts_ms is None:
        raise ValueError("No book timestamp found in strategy.jsonl")

    book_time = _safe_ms_to_datetime(pd.to_numeric(book_ts_ms, errors="coerce"))
    base_time = book_time.min()

    up_bid = pd.to_numeric(df.get("up_bid"), errors="coerce")
    up_ask = pd.to_numeric(df.get("up_ask"), errors="coerce")

    exec_records = load_jsonl(execution_path)
    order_info: dict[int, dict] = {}
    buy_points = []
    sell_points = []
    cancel_points = []

    for rec in exec_records:
        event = rec.get("event")
        if event == "place_received":
            req = rec.get("request") or {}
            client_id = req.get("client_order_id")
            if client_id is not None:
                order_info[int(client_id)] = {
                    "outcome": req.get("outcome"),
                    "side": req.get("side"),
                    "limit_price": req.get("limit_price"),
                }
            continue

        if event in {"report_fill", "fill_applied"}:
            report = rec.get("report") or {}
            client_id = report.get("client_order_id")
            info = order_info.get(int(client_id)) if client_id is not None else None
            outcome = (info or {}).get("outcome")
            side = (info or {}).get("side")
            if outcome and str(outcome).lower() == "up":
                price = report.get("avg_price")
                if price is None:
                    price = (info or {}).get("limit_price")
                ts_ms = rec.get("ts_ms") or report.get("ts_ms")
                if ts_ms is not None and price is not None:
                    dt = _safe_ms_to_datetime(pd.Series([ts_ms])).iloc[0]
                    if pd.notna(dt):
                        elapsed = (dt - base_time).total_seconds()
                        if side and str(side).lower() == "buy":
                            buy_points.append((elapsed, price))
                        elif side and str(side).lower() == "sell":
                            sell_points.append((elapsed, price))
            continue

        if event in {"report_canceled", "cancel_requested"}:
            obj = rec.get("report") or rec.get("cancel") or {}
            client_id = obj.get("client_order_id")
            info = order_info.get(int(client_id)) if client_id is not None else None
            outcome = (info or {}).get("outcome")
            if outcome and str(outcome).lower() == "up":
                price = (info or {}).get("limit_price")
                ts_ms = rec.get("ts_ms") or obj.get("ts_ms")
                if ts_ms is not None and price is not None:
                    dt = _safe_ms_to_datetime(pd.Series([ts_ms])).iloc[0]
                    if pd.notna(dt):
                        elapsed = (dt - base_time).total_seconds()
                        cancel_points.append((elapsed, price))
            continue

    out_dir = Path(args.out) / "combined" / market_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    elapsed_book = (book_time - base_time).dt.total_seconds()
    ax.plot(elapsed_book, up_bid, label="up_bid", linewidth=1)
    ax.plot(elapsed_book, up_ask, label="up_ask", linewidth=1)

    if buy_points:
        xb, yb = zip(*buy_points)
        ax.scatter(xb, yb, color="green", s=30, label="buy fills")
    if sell_points:
        xs, ys = zip(*sell_points)
        ax.scatter(xs, ys, color="red", s=30, label="sell fills")
    if cancel_points:
        xc, yc = zip(*cancel_points)
        ax.scatter(xc, yc, color="gray", s=40, marker="x", label="cancels")

    ax.set_title("Up orderbook + fills + cancels")
    ax.set_ylabel("price")
    _format_price_axis(ax)
    _format_time_axis(ax, elapsed_book)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        fig.savefig(out_dir / f"up_book_fills_cancels.{fmt}", dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved plot to {out_dir}")


if __name__ == "__main__":
    main()
