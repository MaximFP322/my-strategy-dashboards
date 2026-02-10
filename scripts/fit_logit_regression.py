#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _get_path(obj, path):
    if path is None:
        return None
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def main():
    parser = argparse.ArgumentParser(
        description="Fit logit(y) = a + b*z for orderbook mid vs delta/scale."
    )
    parser.add_argument("path", type=Path, help="Path to JSONL (strategy or telemetry)")
    parser.add_argument("--telemetry", action="store_true", help="Use telemetry defaults")
    parser.add_argument("--z-path", default=None, help="Dot path for precomputed z")
    parser.add_argument("--delta-path", default=None, help="Dot path for delta")
    parser.add_argument("--scale-path", default=None, help="Dot path for scale")
    parser.add_argument(
        "--scale-const", type=float, default=None, help="Constant scale (if no scale path)"
    )
    parser.add_argument("--up-bid-path", default=None, help="Dot path for up bid")
    parser.add_argument("--up-ask-path", default=None, help="Dot path for up ask")
    parser.add_argument("--down-bid-path", default=None, help="Dot path for down bid")
    parser.add_argument("--down-ask-path", default=None, help="Dot path for down ask")
    parser.add_argument("--min-points", type=int, default=30, help="Minimum points to fit")
    args = parser.parse_args()

    if args.telemetry:
        args.z_path = args.z_path or "btc.z_delta"
        args.delta_path = args.delta_path or "btc.delta"
        args.up_bid_path = args.up_bid_path or "up_l1.bid_p"
        args.up_ask_path = args.up_ask_path or "up_l1.ask_p"
        args.down_bid_path = args.down_bid_path or "down_l1.bid_p"
        args.down_ask_path = args.down_ask_path or "down_l1.ask_p"
    else:
        args.delta_path = args.delta_path or "delta"
        args.up_bid_path = args.up_bid_path or "up_bid"
        args.up_ask_path = args.up_ask_path or "up_ask"
        args.down_bid_path = args.down_bid_path or "down_bid"
        args.down_ask_path = args.down_ask_path or "down_ask"

    use_scale_const = args.scale_const is not None
    scale_path = args.scale_path
    z_path = args.z_path

    zs = []
    ys = []
    skipped = 0

    with args.path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            z = None
            if z_path:
                z = _as_float(_get_path(r, z_path))
                if z is None:
                    skipped += 1
                    continue
            else:
                delta = _as_float(_get_path(r, args.delta_path))
                if delta is None:
                    skipped += 1
                    continue

                if use_scale_const:
                    scale = args.scale_const
                else:
                    scale = _as_float(_get_path(r, scale_path))
                    if scale is None:
                        skipped += 1
                        continue
                z = delta / (scale + 1e-12)

            up_bid = _as_float(_get_path(r, args.up_bid_path))
            up_ask = _as_float(_get_path(r, args.up_ask_path))
            down_bid = _as_float(_get_path(r, args.down_bid_path))
            down_ask = _as_float(_get_path(r, args.down_ask_path))
            if None in (up_bid, up_ask, down_bid, down_ask):
                skipped += 1
                continue

            up_mid = 0.5 * (up_bid + up_ask)
            down_mid = 0.5 * (down_bid + down_ask)
            y = 0.5 * (up_mid + (1 - down_mid))

            zs.append(z)
            ys.append(logit(y))

    zs = np.array(zs, dtype=float)
    ys = np.array(ys, dtype=float)

    if len(zs) < args.min_points:
        print(f"Not enough points ({len(zs)}) after filtering. Skipped={skipped}")
        return

    var_z = np.var(zs)
    if var_z == 0:
        print("Zero variance in z; cannot fit.")
        return

    b = np.cov(zs, ys, bias=True)[0, 1] / var_z
    a = ys.mean() - b * zs.mean()
    pred = a + b * zs
    ss_res = np.sum((ys - pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    print(f"points={len(zs)} skipped={skipped}")
    print(f"a={a} b={b} r2={r2}")


if __name__ == "__main__":
    main()
