"""Exact-output tests for the geographic freeze path in pilot/runner.py."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import georeset_text_label_benchmark.pilot.runner as runner
from georeset_text_label_benchmark.output_schema import LABELLED_SCHEMA
from georeset_text_label_benchmark.pilot.geo_sampling import (
    h3_cell_of,
    h3_centre_of,
    sample_id_of,
    select_geographic_sample,
)
from georeset_text_label_benchmark.pilot.runner import SELECTION_METHOD, freeze_sample

LICENSE = "OpenStreetMap ODbL-1.0; European Environment Agency CC-BY-4.0"
TEXT_METHOD = "Exact pinned EUNIS English name, newline, exact EEA description."


def _candidate_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pilot_data/eunis_candidate_labels.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pinned EUNIS candidate table is missing")


def _candidates() -> list[dict[str, str]]:
    with _candidate_file().open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_of_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _outside_row() -> dict[str, Any]:
    sentence = "Snow-bed sentence."
    return {
        "source_pbf": "austria-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": 900001,
        "description_identity": "description-outside",
        "tag_key": "description",
        "sentence_index": 0,
        "sentence": sentence,
        "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
        "language_code": "eng",
        "eunis_code": "R41",
        "eunis_name": "Snow-bed vegetation",
        "eunis_overlap_percentage": 80.0,
        "eunis_source_version": "maps-v1",
        "decision": "no",
        "bbox_min_x": 10.0,
        "bbox_min_y": 47.0,
        "bbox_max_x": 10.2,
        "bbox_max_y": 47.2,
    }


def _write_pool(directory: Path, extra: Sequence[dict[str, Any]] = ()) -> tuple[Path, Path]:
    """Every candidate code gets one yes and one no sentence at a random place on Earth."""
    rng = random.Random(7)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(_candidates()):
        for offset, decision in enumerate(("yes", "no")):
            lat = rng.uniform(-60.0, 70.0)
            lon = rng.uniform(-170.0, 170.0)
            sentence = f"Pool sentence {index} {decision}."
            rows.append(
                {
                    "source_pbf": "test-latest.osm.pbf",
                    "osm_type": "way",
                    "osm_id": index * 2 + offset + 1,
                    "osm_url": None,
                    "description_identity": f"description-{index}-{decision}",
                    "tag_key": "description",
                    "sentence_index": 0,
                    "sentence": sentence,
                    "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
                    "language_code": "eng",
                    "eunis_code": candidate["eunis_code"],
                    "eunis_name": candidate["eunis_name"],
                    "eunis_overlap_percentage": 80.0,
                    "eunis_source_version": "maps-v1",
                    "decision": decision,
                    "bbox_min_x": lon - 0.1,
                    "bbox_min_y": lat - 0.1,
                    "bbox_max_x": lon + 0.1,
                    "bbox_max_y": lat + 0.1,
                }
            )
    rows.extend(extra)
    rows = [{key: row.get(key) for key in LABELLED_SCHEMA.names} for row in rows]
    labelled = directory / "labelled-eunis.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=LABELLED_SCHEMA), labelled)
    manifest = directory / "pipeline-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "computation_commit": "a" * 40,
                "validation_commit": "b" * 40,
                "input_snapshots": {"shared_input_revision": "shared-revision"},
                "artifact_sha256": {"labelled-eunis.parquet": _sha256(labelled)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return labelled, manifest


def _labelled_rows(labelled: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(
        labelled, columns=list(LABELLED_SCHEMA.names)
    ).to_pylist()
    return rows


@pytest.fixture
def pool(tmp_path: Path) -> tuple[Path, Path, Path]:
    labelled, manifest = _write_pool(tmp_path)
    return labelled, manifest, _candidate_file()


def _freeze(pool: tuple[Path, Path, Path], output: Path, **options: Any) -> dict[str, Any]:
    labelled, manifest, candidate = pool
    return freeze_sample(labelled, candidate, manifest, output, **options)


def test_freeze_selects_with_the_requested_per_group_and_seed(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output, per_group=5, seed=7)

    sample = _read_json(output / "frozen_sample.json")
    reference = select_geographic_sample(
        _labelled_rows(pool[0]),
        cell_of=h3_cell_of,
        centre_of=h3_centre_of,
        per_group=5,
        seed=7,
    )
    assert sample["selected_rows"] == [{**row, "sample_id": sample_id_of(row)} for row in reference]
    assert sample["selection"]["seed"] == 7
    assert sample["selection"]["per_group"] == 5
    assert sample["selection"]["sample_size"] == 10
    assert sample["selection"]["decision_counts"] == {"no": 5, "yes": 5}


def test_freeze_summary_reports_the_exact_sample_and_candidate_counts(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    summary = _freeze(pool, output)

    sample = _read_json(output / "frozen_sample.json")
    assert summary == {
        "output_dir": str(output),
        "sample_size": 100,
        "candidate_count": 158,
        "sample_ids_sha256": sample["selection"]["sample_ids_sha256"],
        "frozen_sample_sha256": _sha256(output / "frozen_sample.json"),
    }


def test_frozen_sample_declares_its_format_and_candidate_block_exactly(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)

    sample = _read_json(output / "frozen_sample.json")
    assert set(sample) == {
        "format_version",
        "source",
        "source_coverage",
        "selection",
        "candidate_labels",
        "selected_rows",
    }
    assert sample["format_version"] == 2
    assert sample["candidate_labels"] == {
        "file": "candidate_labels.csv",
        "sha256": _sha256(_candidate_file()),
        "count": 158,
        "codes": sorted(row["eunis_code"] for row in _candidates()),
        "text_method": TEXT_METHOD,
    }


def test_frozen_source_names_every_pinned_input_exactly(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    labelled, manifest, _candidate = pool
    output = tmp_path / "run"
    _freeze(pool, output)

    sample = _read_json(output / "frozen_sample.json")
    assert sample["source"] == {
        "file": "labelled-eunis.parquet",
        "sha256": _sha256(labelled),
        "pipeline_manifest": "pipeline-manifest.json",
        "pipeline_manifest_sha256": _sha256(manifest),
        "computation_commit": "a" * 40,
        "validation_commit": "b" * 40,
        "input_snapshots": {"shared_input_revision": "shared-revision"},
        "license": LICENSE,
    }


def test_source_coverage_counts_the_candidate_vocabulary_rows_only(tmp_path: Path) -> None:
    labelled, manifest = _write_pool(tmp_path, extra=[_outside_row()])
    output = tmp_path / "run"
    freeze_sample(labelled, _candidate_file(), manifest, output)

    sample = _read_json(output / "frozen_sample.json")
    assert sample["source_coverage"] == {
        "row_count": 316,
        "unique_sentence_hash_count": 316,
        "unique_polygon_count": 316,
        "observed_eunis_class_count": 158,
        "rows_outside_candidate_vocabulary": 1,
    }


def test_selection_block_reports_exact_counts_and_id_checksum(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)

    sample = _read_json(output / "frozen_sample.json")
    ids = [row["sample_id"] for row in sample["selected_rows"]]
    assert sample["selection"] == {
        "method": SELECTION_METHOD,
        "seed": 42,
        "per_group": 50,
        "sample_size": 100,
        "decision_counts": {"no": 50, "yes": 50},
        "h3_resolution": 3,
        "cell_count": 100,
        "sample_ids_sha256": _digest_of_json(ids),
        "unique_sentence_hash_count": 100,
        "unique_polygon_count": 100,
    }


def test_manifest_records_format_inputs_and_output_checksums_exactly(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    labelled, pipeline_manifest, _candidate = pool
    output = tmp_path / "run"
    _freeze(pool, output)

    sample = _read_json(output / "frozen_sample.json")
    assert _read_json(output / "manifest.json") == {
        "format_version": 2,
        "method": SELECTION_METHOD,
        "seed": 42,
        "per_group": 50,
        "sample_size": 100,
        "sample_ids_sha256": sample["selection"]["sample_ids_sha256"],
        "source_sha256": {
            "labelled-eunis.parquet": _sha256(labelled),
            "pipeline-manifest.json": _sha256(pipeline_manifest),
        },
        "outputs_sha256": {
            "frozen_sample.json": _sha256(output / "frozen_sample.json"),
            "candidate_labels.csv": _sha256(output / "candidate_labels.csv"),
        },
    }


def test_freeze_creates_missing_parent_directories(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "missing" / "nested" / "run"
    _freeze(pool, output)

    assert sorted(path.name for path in output.iterdir()) == [
        "candidate_labels.csv",
        "frozen_sample.json",
        "manifest.json",
    ]


def test_freeze_refuses_a_directory_created_after_the_existence_check(
    pool: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    original_validate = runner._validate_freeze_inputs

    def create_output_then_validate(
        labelled_parquet: Path,
        candidate_csv: Path,
        pipeline_manifest: Path,
        expected_candidate_sha256: str | None,
    ) -> tuple[str, str, Any]:
        output.mkdir()
        return original_validate(
            labelled_parquet, candidate_csv, pipeline_manifest, expected_candidate_sha256
        )

    monkeypatch.setattr(runner, "_validate_freeze_inputs", create_output_then_validate)

    with pytest.raises(
        FileExistsError, match=rf"^\[Errno 17\] File exists: '{re.escape(str(output))}'$"
    ):
        _freeze(pool, output)
    assert list(output.iterdir()) == []
