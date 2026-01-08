from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting.io import filter_records, load_jsonl  # noqa: E402
from plotting.plots import generate_plots  # noqa: E402
from plotting.processing import maybe_downsample, normalize_records  # noqa: E402
from plotting.summary import summarize, write_summary  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot strategy debug logs from JSONL output.",
    )
    parser.add_argument(
        "--input",
        default="output/json/strategy_debug.jsonl",
        help="Path to JSONL log file",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Optional positional path to JSONL log file",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name to filter (e.g., fair_edge or updown)",
    )
    parser.add_argument(
        "--market",
        dest="market_slug",
        default=None,
        help="Market slug to filter (optional)",
    )
    parser.add_argument(
        "--out",
        default="output/plots",
        help="Output root directory for plots",
    )
    parser.add_argument(
        "--formats",
        default="png,svg",
        help="Comma-separated list of plot formats",
    )
    parser.add_argument(
        "--downsample-rule",
        default="50ms",
        help="Pandas resample rule for downsampling (set to 'none' to disable)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100_000,
        help="Maximum number of points before downsampling",
    )
    parser.add_argument(
        "--summary",
        nargs="?",
        const="summary.json",
        default=None,
        help="Write summary JSON (default summary.json in output dir)",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=1.0,
        help="Scale factor for figure size (e.g., 1.5 for larger plots)",
    )
    parser.add_argument(
        "--xscale",
        type=float,
        default=1.0,
        help="Horizontal scale factor for figure width",
    )
    parser.add_argument(
        "--yscale",
        type=float,
        default=1.0,
        help="Vertical scale factor for figure height",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved images",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.input_path:
        args.input = args.input_path

    formats = tuple(
        fmt.strip() for fmt in args.formats.split(",") if fmt.strip()
    )
    if not formats:
        formats = ("png",)

    downsample_rule = args.downsample_rule
    if downsample_rule.lower() == "none":
        downsample_rule = None

    records = load_jsonl(args.input)
    filtered = filter_records(
        records, strategy=args.strategy, market_slug=args.market_slug
    )
    df = normalize_records(filtered)
    df = maybe_downsample(
        df, max_points=args.max_points, rule=downsample_rule
    )

    market_label = args.market_slug or "all"
    out_dir = Path(args.out) / args.strategy / market_label
    created = generate_plots(
        df,
        out_dir,
        formats=formats,
        figscale=args.figscale,
        xscale=args.xscale,
        yscale=args.yscale,
        dpi=args.dpi,
    )

    if not created:
        raise ValueError("No plots created. Check input columns and filters.")

    if args.summary is not None:
        summary = summarize(df)
        summary_path = Path(args.summary)
        if not summary_path.is_absolute():
            summary_path = out_dir / summary_path
        write_summary(summary, summary_path)

    print(f"Created {len(created)} plot files in {out_dir}")


if __name__ == "__main__":
    main()
