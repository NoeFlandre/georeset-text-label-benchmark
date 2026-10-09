"""Command-line interface for freezing IDs and running the LFM2.5 DSpark pilot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from georeset_text_label_benchmark.pilot.dspark_runner import run_dspark_pilot
from georeset_text_label_benchmark.pilot.protocol import SAMPLE_SEED, SAMPLE_SIZE
from georeset_text_label_benchmark.pilot.runner import freeze_sample

PILOT_RUN_DIR = Path("artifacts/pilot-100-seed42")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="georeset-pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze the geographic yes/no sample")
    freeze.add_argument("--labelled-parquet", type=Path, required=True)
    freeze.add_argument("--pipeline-manifest", type=Path, required=True)
    freeze.add_argument(
        "--candidate-csv", type=Path, default=Path("pilot_data/eunis_candidate_labels.csv")
    )
    freeze.add_argument("--output-dir", type=Path, default=PILOT_RUN_DIR)
    freeze.add_argument("--per-group", type=int, default=SAMPLE_SIZE // 2)
    freeze.add_argument("--seed", type=int, default=SAMPLE_SEED)
    dspark = commands.add_parser(
        "run-dspark", help="predict one EUNIS code per row with pinned LFM2.5 + DSpark"
    )
    dspark.add_argument("--run-dir", type=Path, default=PILOT_RUN_DIR)
    dspark.add_argument("--output-dir", type=Path)
    dspark.add_argument("--model-cache", type=Path, default=Path(".cache/model-dspark"))
    dspark.add_argument("--model-cache-seed", type=Path)
    dspark.add_argument("--computation-commit", required=True)
    dspark.add_argument("--validation-commit", required=True)
    smoke = commands.add_parser(
        "run-dspark-smoke", help="run a bounded eight-row LFM2.5 + DSpark readiness smoke"
    )
    smoke.add_argument("--run-dir", type=Path, default=PILOT_RUN_DIR)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--model-cache", type=Path, default=Path(".cache/model-dspark"))
    smoke.add_argument("--model-cache-seed", type=Path)
    smoke.add_argument("--computation-commit", required=True)
    smoke.add_argument("--validation-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _freeze_sample_command(args) if args.command == "freeze" else _run_dspark_command(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "run-dspark-smoke" and not result["smoke_gate"]["passed"]:
        return 1
    return 0


def _freeze_sample_command(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze the geographic yes/no sample before inference."""
    return freeze_sample(
        args.labelled_parquet,
        args.candidate_csv,
        args.pipeline_manifest,
        args.output_dir,
        per_group=args.per_group,
        seed=args.seed,
    )


def _run_dspark_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run the DSpark pilot or bounded smoke for the selected subcommand."""
    return run_dspark_pilot(
        args.run_dir,
        output_dir=args.output_dir,
        model_cache_dir=args.model_cache,
        model_cache_seed_dir=args.model_cache_seed,
        computation_commit=args.computation_commit,
        validation_commit=args.validation_commit,
        smoke=args.command == "run-dspark-smoke",
    )
