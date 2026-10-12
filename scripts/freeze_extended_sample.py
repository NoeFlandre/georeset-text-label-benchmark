"""Freeze the published geographic sample extended to more rows per group.

The published 100-row sample is kept unchanged; each group is extended by maximin
spacing. Writes the same three frozen inputs as ``georeset-pilot freeze``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from georeset_text_label_benchmark.pilot.protocol import SAMPLE_SEED
from georeset_text_label_benchmark.pilot.runner import freeze_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled-parquet", type=Path, required=True)
    parser.add_argument("--pipeline-manifest", type=Path, required=True)
    parser.add_argument("--base-sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument(
        "--candidate-csv", type=Path, default=Path("pilot_data/eunis_candidate_labels.csv")
    )
    args = parser.parse_args()
    summary = freeze_sample(
        args.labelled_parquet,
        args.candidate_csv,
        args.pipeline_manifest,
        args.output_dir,
        per_group=args.per_group,
        seed=args.seed,
        extend_from=args.base_sample,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
