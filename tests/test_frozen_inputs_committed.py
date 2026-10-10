"""The frozen 100-row inputs that the local DSpark run reads are committed."""

from __future__ import annotations

from pathlib import Path

from georeset_text_label_benchmark.pilot import dspark
from georeset_text_label_benchmark.pilot.runner import read_frozen_pilot_inputs

RUN_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "pilot-100-seed42"


def test_committed_frozen_files_are_present() -> None:
    for name in ("frozen_sample.json", "candidate_labels.csv", "manifest.json"):
        assert (RUN_DIR / name).is_file(), name


def test_committed_frozen_inputs_load_and_match_the_pinned_sample() -> None:
    sample, rows, candidates = read_frozen_pilot_inputs(RUN_DIR)

    assert len(rows) == 100
    assert sum(row["decision"] == "yes" for row in rows) == 50
    assert sum(row["decision"] == "no" for row in rows) == 50
    assert sample["selection"]["sample_ids_sha256"] == dspark.EXPECTED_SAMPLE_IDS_SHA256
    assert candidates
