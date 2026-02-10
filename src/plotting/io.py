from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {lineno}: {exc}") from exc
            if isinstance(record, dict):
                record.setdefault("__line__", lineno)
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def filter_records(
    records: Iterable[dict],
    *,
    strategy: str,
    market_slug: str | None = None,
) -> list[dict]:
    filtered: list[dict] = []
    execution_events = {
        "place_received",
        "report_submitted",
        "report_accepted",
        "report_rejected",
        "report_fill",
        "report_final",
        "cancel_requested",
        "report_canceled",
        "report_cancel_rejected",
        "report_cancel_failed",
        "fill_applied",
    }
    for record in records:
        if strategy == "portfolio":
            event = record.get("event")
            strategy_value = record.get("strategy")
            if strategy_value is not None and strategy_value != "portfolio":
                continue
            if event is not None and event not in {"snapshot", "settled"}:
                continue
        elif strategy == "execution":
            event = record.get("event")
            if event is None or event not in execution_events:
                continue
        else:
            if record.get("strategy") != strategy:
                continue
        if market_slug:
            slug = record.get("market_slug")
            if slug is None:
                market = record.get("market")
                if isinstance(market, dict):
                    slug = market.get("slug")
            if slug != market_slug:
                continue
        filtered.append(record)

    if not filtered:
        message = f"No records match strategy={strategy!r}"
        if market_slug:
            message += f" and market_slug={market_slug!r}"
        raise ValueError(message)

    return filtered
