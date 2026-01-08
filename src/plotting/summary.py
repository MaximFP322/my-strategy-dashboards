from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    summary: dict[str, object] = {
        "rows": int(len(df)),
    }

    if "decision" in df.columns:
        summary["decisions"] = (
            df["decision"].fillna("<null>").value_counts().to_dict()
        )

    if "exposure" in df.columns:
        summary["exposure"] = (
            df["exposure"].fillna("<null>").value_counts().to_dict()
        )

    if "orders" in df.columns:
        orders_count = 0
        for value in df["orders"]:
            if isinstance(value, list):
                orders_count += len(value)
        summary["orders_total"] = orders_count

    return summary


def write_summary(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
