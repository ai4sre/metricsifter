"""Command-line interface for MetricSifter (stdlib ``argparse`` only).

Usage::

    metricsifter run INPUT.csv [--output OUT.csv] [--report REPORT.json] ...

The ``main(argv)`` entry point returns an exit code (0 success, 2 input error)
instead of calling ``sys.exit`` directly, so it can be driven from tests without
subprocesses.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import pandas as pd

from metricsifter.sifter import Sifter

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metricsifter",
        description="Feature reduction for multivariate time series data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Sift metrics from a CSV file.")
    run.add_argument("input", help="Input CSV file of time series (columns are metrics).")
    run.add_argument("--output", "-o", default=None, help="Output CSV path for the sifted metrics (default: stdout).")
    run.add_argument("--report", default=None, help="Path to write the SiftResult diagnostic report as JSON.")
    run.add_argument(
        "--penalty-adjust",
        type=_penalty_adjust_value,
        default=2.0,
        help="Penalty adjustment factor (default: 2.0), or 'auto' for stability selection.",
    )
    run.add_argument(
        "--bandwidth",
        type=_bandwidth_value,
        default=2.5,
        help="KDE bandwidth for segmentation (default: 2.5); also accepts 'scott', 'silverman' or 'auto'.",
    )
    run.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Seed for the 'auto' bandwidth bootstrap (default: nondeterministic).",
    )
    run.add_argument(
        "--search-method",
        default="pelt",
        choices=["pelt", "binseg", "bottomup"],
        help="Change-point search method (default: pelt).",
    )
    run.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (default: 1).")
    run.add_argument(
        "--index-col",
        default="none",
        help="Column to use as the row index (int position or name). By default every column is "
        "treated as a metric; pass e.g. '--index-col 0' when the CSV has a time/index column.",
    )
    run.add_argument(
        "--parse-dates", action="store_true", help="Parse the index column (see --index-col) as datetimes."
    )
    return parser


def _penalty_adjust_value(value: str) -> float | str:
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a float or 'auto', got {value!r}")


def _bandwidth_value(value: str) -> float | str:
    if value in {"auto", "scott", "silverman"}:
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a float, 'scott', 'silverman' or 'auto', got {value!r}")


def _resolve_index_col(value: str) -> int | str | None:
    if value is None or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _run(args: argparse.Namespace) -> int:
    index_col = _resolve_index_col(args.index_col)
    try:
        data = pd.read_csv(
            args.input,
            index_col=index_col,
            parse_dates=bool(args.parse_dates),
        )
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
        print(f"error: failed to read CSV {args.input!r}: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if data.shape[1] == 0:
        print(f"error: input {args.input!r} has no metric columns.", file=sys.stderr)
        return EXIT_INPUT_ERROR

    sifter = Sifter(
        search_method=args.search_method,
        penalty_adjust=args.penalty_adjust,
        bandwidth=args.bandwidth,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    result = sifter.sift(data)

    out_df = result.data if result.data is not None else pd.DataFrame()
    if args.output:
        out_df.to_csv(args.output)
    else:
        out_df.to_csv(sys.stdout)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fp:
            fp.write(result.to_json(indent=2))

    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover - argparse guards this
    return EXIT_INPUT_ERROR  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
