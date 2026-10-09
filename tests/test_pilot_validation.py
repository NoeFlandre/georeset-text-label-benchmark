"""Guards for the shared pilot validators: metrics, CLI exit status and frozen-input checks."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from georeset_text_label_benchmark.pilot import cli, metrics, runner


def _candidate_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pilot_data/eunis_candidate_labels.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pinned EUNIS candidate table is missing")


def _assert_error(action: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        action()
    assert str(error.value) == message


def test_single_label_report_counts_invalid_outputs_as_misses() -> None:
    report = metrics.single_label_report(["A", "B", "A"], ["A", None, "C"], ["A", "B", "C"])

    assert report["prediction_count"] == 2
    assert report["invalid_output_count"] == 1
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["top1_accuracy"] == pytest.approx(1 / 3)
    assert report["macro_f1_all_candidates"] == pytest.approx(2 / 9)


def test_single_label_class_breakdown_reports_precision_and_recall() -> None:
    rows = metrics.single_label_class_breakdown(["A", "A", "B"], ["A", "B", "B"], ["A", "B"])

    by_code = {row["eunis_code"]: row for row in rows}
    assert by_code["A"]["top1_correct"] == 1
    assert by_code["A"]["top1_precision"] == pytest.approx(1.0)
    assert by_code["A"]["top1_recall"] == pytest.approx(0.5)
    assert by_code["B"]["top1_correct"] == 1
    assert by_code["B"]["top1_precision"] == pytest.approx(0.5)
    assert by_code["B"]["top1_recall"] == pytest.approx(1.0)


@pytest.mark.parametrize("candidates", [[], ["A", "A"]])
def test_single_label_report_requires_unique_non_empty_candidates(candidates: list[str]) -> None:
    _assert_error(
        lambda: metrics.single_label_report(["A"], ["A"], candidates),
        "candidate codes must be non-empty and unique",
    )


def _smoke_arguments() -> list[str]:
    return [
        "run-dspark-smoke",
        "--output-dir",
        "out",
        "--computation-commit",
        "a" * 40,
        "--validation-commit",
        "b" * 40,
    ]


def test_smoke_exit_status_follows_the_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "run_dspark_pilot", lambda *_a, **_k: {"smoke_gate": {"passed": False}}
    )
    assert cli.main(_smoke_arguments()) == 1
    monkeypatch.setattr(cli, "run_dspark_pilot", lambda *_a, **_k: {"smoke_gate": {"passed": True}})
    assert cli.main(_smoke_arguments()) == 0
    capsys.readouterr()


def test_full_run_exit_status_does_not_read_the_smoke_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "run_dspark_pilot", lambda *_a, **_k: {"run_dir": "out"})
    arguments = [
        "run-dspark",
        "--computation-commit",
        "a" * 40,
        "--validation-commit",
        "b" * 40,
    ]

    assert cli.main(arguments) == 0
    capsys.readouterr()


def test_taxonomy_header_must_match_the_pinned_schema() -> None:
    runner._validate_taxonomy_header(list(runner.CANDIDATE_FIELDS))

    _assert_error(
        lambda: runner._validate_taxonomy_header(["eunis_code"]),
        "candidate CSV schema does not match the pinned pilot schema",
    )


@pytest.mark.parametrize("commit", ["a" * 39, "A" * 40, "g" * 40, "X" + "a" * 39])
def test_commit_must_be_forty_lowercase_hex_characters(commit: str) -> None:
    _assert_error(
        lambda: runner._validate_commit(commit, "computation_commit"),
        "computation_commit must be a 40-character lowercase Git commit SHA",
    )
    runner._validate_commit("a" * 40, "computation_commit")


def test_frozen_values_must_be_unique() -> None:
    runner._require_unique_frozen_values(["a", "b"])

    _assert_error(
        lambda: runner._require_unique_frozen_values(["a", "a"]),
        "frozen sample must have unique IDs, sentence hashes, and polygons",
    )


def test_frozen_row_count_and_id_checksum_are_enforced() -> None:
    runner._validate_frozen_row_count([{}, {}], {"sample_size": 2})
    _assert_error(
        lambda: runner._validate_frozen_row_count([{}], {"sample_size": 2}),
        "frozen sample row count does not match its declaration",
    )

    ids = ["x", "y"]
    runner._validate_frozen_ids(ids, {"sample_ids_sha256": runner._sha256_json(ids)})
    _assert_error(
        lambda: runner._validate_frozen_ids(ids, {"sample_ids_sha256": "0" * 64}),
        "frozen sample ID checksum mismatch",
    )


def test_candidate_code_list_must_match_the_frozen_protocol() -> None:
    candidates: list[dict[str, Any]] = [{"eunis_code": "A"}, {"eunis_code": "B"}]
    runner._validate_candidate_code_list(candidates, {"candidate_labels": {"codes": ["A", "B"]}})

    _assert_error(
        lambda: runner._validate_candidate_code_list(
            candidates, {"candidate_labels": {"codes": ["B", "A"]}}
        ),
        "candidate code list does not match the frozen protocol",
    )


def test_candidate_rows_reject_a_changed_candidate_file(tmp_path: Path) -> None:
    shutil.copyfile(_candidate_file(), tmp_path / "candidate_labels.csv")
    sample = {"candidate_labels": {"file": "candidate_labels.csv", "sha256": "0" * 64}}

    _assert_error(
        lambda: runner._candidate_rows(tmp_path, sample),
        "candidate label file checksum mismatch",
    )
