"""Command-line entry point for the pinned overlap pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from georeset_text_label_benchmark.pipeline import run_pipeline
from georeset_text_label_benchmark.source import DescriptionSource


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="georeset-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="compute overlap from pinned public Hub snapshots")
    run.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/description-eunis-overlap"),
        help="new directory for overlap.parquet, summary.json, and manifest.json",
    )
    run.add_argument(
        "--computation-commit",
        required=True,
        help="40-character Git commit SHA of the code used to generate the artifacts",
    )
    run.add_argument(
        "--validation-commit",
        required=True,
        help="40-character Git commit SHA of the code used to validate the artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_pipeline(
        DescriptionSource.from_hub(),
        args.output,
        computation_commit=args.computation_commit,
        validation_commit=args.validation_commit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
