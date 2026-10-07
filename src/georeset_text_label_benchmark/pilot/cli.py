"""Command-line interface for freezing IDs and running the E5 pilot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from georeset_text_label_benchmark.pilot.dspark_runner import run_dspark_pilot
from georeset_text_label_benchmark.pilot.protocol import (
    BATCH_SIZE,
    MAX_LENGTH,
    OVERLAP_PARQUET_SHA256,
    OVERLAP_REVISION,
    SAMPLE_SEED,
    SAMPLE_SIZE,
)
from georeset_text_label_benchmark.pilot.runner import freeze_sample, run_pilot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="georeset-pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze selected rows before inference")
    freeze.add_argument("--source-parquet", type=Path, required=True)
    freeze.add_argument(
        "--candidate-csv", type=Path, default=Path("pilot_data/eunis_candidate_labels.csv")
    )
    freeze.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/e5-small-100-seed42"),
    )
    freeze.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    freeze.add_argument("--seed", type=int, default=SAMPLE_SEED)
    run = commands.add_parser("run", help="run frozen zero-shot candidate ranking")
    run.add_argument("--run-dir", type=Path, default=Path("artifacts/e5-small-100-seed42"))
    run.add_argument("--model-cache", type=Path, default=Path(".cache/huggingface"))
    run.add_argument("--computation-commit", required=True)
    run.add_argument("--validation-commit", required=True)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--max-length", type=int, default=MAX_LENGTH)
    dspark = commands.add_parser(
        "run-dspark", help="predict one EUNIS code per row with pinned LFM2.5 + DSpark"
    )
    dspark.add_argument("--run-dir", type=Path, default=Path("artifacts/e5-small-100-seed42"))
    dspark.add_argument("--output-dir", type=Path)
    dspark.add_argument("--model-cache", type=Path, default=Path(".cache/model-dspark"))
    dspark.add_argument("--computation-commit", required=True)
    dspark.add_argument("--validation-commit", required=True)
    smoke = commands.add_parser(
        "run-dspark-smoke", help="run a bounded eight-row LFM2.5 + DSpark readiness smoke"
    )
    smoke.add_argument("--run-dir", type=Path, default=Path("artifacts/e5-small-100-seed42"))
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--model-cache", type=Path, default=Path(".cache/model-dspark"))
    smoke.add_argument("--computation-commit", required=True)
    smoke.add_argument("--validation-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        result = _freeze_sample_command(args)
    elif args.command == "run":
        result = _run_e5_command(args)
    else:
        result = _run_dspark_command(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "run-dspark-smoke" and not result["smoke_gate"]["passed"]:
        return 1
    return 0


def _freeze_sample_command(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze the requested source rows before inference."""
    return freeze_sample(
        args.source_parquet,
        args.candidate_csv,
        args.output_dir,
        expected_source_sha256=OVERLAP_PARQUET_SHA256,
        source_revision=OVERLAP_REVISION,
        size=args.sample_size,
        seed=args.seed,
    )


def _run_e5_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen E5 ranking pilot."""
    return run_pilot(
        args.run_dir,
        args.model_cache,
        computation_commit=args.computation_commit,
        validation_commit=args.validation_commit,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


def _run_dspark_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run the DSpark pilot or bounded smoke for the selected subcommand."""
    return run_dspark_pilot(
        args.run_dir,
        output_dir=args.output_dir,
        model_cache_dir=args.model_cache,
        computation_commit=args.computation_commit,
        validation_commit=args.validation_commit,
        smoke=args.command == "run-dspark-smoke",
    )
