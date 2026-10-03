"""Small local end-to-end checks for artifact publication and audit counts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pyarrow import parquet as pq

from georeset_text_label_benchmark.config import DatasetSnapshots
from georeset_text_label_benchmark.errors import DataValidationError
from georeset_text_label_benchmark.models import SourcePartition
from georeset_text_label_benchmark.pipeline import (
    DESCRIPTION_COLUMNS,
    LABEL_COLUMNS,
    POLYGON_COLUMNS,
    run_pipeline,
)

REVISION = "shared-revision"
PBF = "tuvalu-latest.osm.pbf"
SNAPSHOTS = DatasetSnapshots("owner/labels", "label-sha", "owner/eunis", "eunis-sha", REVISION)


class TinySource:
    snapshots = SNAPSHOTS

    def __init__(self, *, empty: bool = False, multi_partition: bool = False) -> None:
        self.partition = SourcePartition(
            "tuvalu-latest.parquet", "labels", "descriptions", "polygons"
        )
        self.rows: dict[str, list[dict[str, Any]]] = {
            "labels": [
                _label("d1", "description", 0, "Assigned habitat sentence.", "yes"),
                _label("d1", "description", 1, "A negative sentence.", "no"),
                _label("d2", "description:en", 0, "Yes without EUNIS.", "yes", language="tgl"),
                _label(
                    "d3", "description", 0, "No sentence list.", "skipped_unsplit", language="tgl"
                ),
                _label("d4", "description", 0, "Unparsed sentence.", "failed"),
            ],
            "descriptions": [
                _description(
                    "d1", "description", 1, ["Assigned habitat sentence.", "A negative sentence."]
                ),
                _description("d2", "description:en", 2, ["Yes without EUNIS."], language="tgl"),
                _description("d3", "description", 3, [], language="tgl"),
                _description("d4", "description", 4, ["Unparsed sentence."]),
            ],
            "polygons": [
                _polygon(1, "T11"),
                _polygon(2, None),
                _polygon(3, None),
                _polygon(4, "T22"),
                _polygon(5, "T33"),
            ],
        }
        if empty:
            self.partitions_value: list[SourcePartition] = []
        else:
            self.partitions_value = [self.partition]
        if multi_partition:
            second = SourcePartition(
                "vanuatu-latest.parquet", "labels-2", "descriptions-2", "polygons-2"
            )
            repeated_text = "Assigned habitat sentence."
            self.rows.update(
                {
                    "labels-2": [_label("d5", "description", 0, repeated_text, "yes")],
                    "descriptions-2": [_description("d5", "description", 6, [repeated_text])],
                    "polygons-2": [_polygon(6, "T11"), _polygon(7, "T44")],
                }
            )
            self.partitions_value.append(second)

    def partitions(self) -> list[SourcePartition]:
        return self.partitions_value

    def read_rows(self, path: str, columns: Sequence[str]) -> list[dict[str, Any]]:
        return [{column: row[column] for column in columns} for row in self.rows[path]]

    def eunis_manifest(self) -> dict[str, Any]:
        return {
            "source_revision": REVISION,
            "eunis_reference_version": "maps-v1",
            "eunis_reference_asset_count": 7,
            "eunis_manifest_sha256": "a" * 64,
        }


def _label(
    identity: str,
    tag: str,
    index: int,
    sentence: str,
    decision: str,
    *,
    language: str = "eng",
) -> dict[str, Any]:
    return {
        "description_identity": identity,
        "tag_key": tag,
        "sentence_index": index,
        "text_sha256": hashlib.sha256(sentence.encode()).hexdigest(),
        "decision": decision,
        "language": language,
        "input_revision": REVISION,
    }


def _description(
    identity: str,
    tag: str,
    osm_id: int,
    sentences: list[str],
    *,
    language: str | None = "eng",
) -> dict[str, Any]:
    return {
        "description_identity": identity,
        "source_pbf": PBF,
        "osm_type": "way",
        "osm_id": osm_id,
        "tag_key": tag,
        "language_code": language,
        "sentence_count": len(sentences),
        "sentences": sentences,
    }


def _polygon(osm_id: int, code: str | None) -> dict[str, Any]:
    return {
        "source_pbf": PBF,
        "osm_type": "way",
        "osm_id": osm_id,
        "eunis_code": code,
        "eunis_name": f"Habitat {code}" if code else None,
        "eunis_overlap_percentage": 80.0 if code else None,
        "eunis_source_version": "maps-v1" if code else None,
    }


def test_run_writes_projected_occurrences_and_stage_report(tmp_path: Path) -> None:
    output = tmp_path / "run"

    result = run_pipeline(TinySource(), output, expected_counts=None)

    rows = pq.read_table(output / "overlap.parquet").to_pylist()
    assert [row["sentence"] for row in rows] == ["Assigned habitat sentence."]
    assert pq.read_schema(output / "overlap.parquet").names == [
        "source_pbf",
        "osm_type",
        "osm_id",
        "description_identity",
        "tag_key",
        "sentence_index",
        "sentence",
        "text_sha256",
        "language_code",
        "eunis_code",
        "eunis_name",
        "eunis_overlap_percentage",
        "eunis_source_version",
    ]
    assert result["stages"] == {
        "polygon_rows": 5,
        "assigned_polygon_rows": 3,
        "polygons_without_description": 1,
        "description_tag_rows": 4,
        "sentence_label_rows": 5,
        "sentence_rows": 4,
        "yes_sentence_rows": 2,
        "no_sentence_rows": 1,
        "failed_sentence_rows": 1,
        "skipped_unsplit_rows": 1,
        "yes_without_eunis_rows": 1,
        "retained_overlap_rows": 1,
        "unique_polygon_keys": 5,
        "unique_description_tag_keys": 4,
        "unique_sentence_occurrences": 5,
        "unique_sentence_hashes": 4,
    }
    assert result["labels_by_decision_and_eunis_assignment"]["yes"] == {
        "assigned": 1,
        "missing": 1,
    }
    assert result["baseline_verification"]["checked"] is False
    assert json.loads((output / "manifest.json").read_text())["output_row_count"] == 1
    assert json.loads((output / "summary.json").read_text()) == result


def test_run_refuses_existing_output_without_touching_it(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(DataValidationError, match="already exists"):
        run_pipeline(TinySource(), output, expected_counts=None)

    assert sentinel.read_text() == "keep"


def test_run_rejects_empty_source_and_leaves_no_published_output(tmp_path: Path) -> None:
    output = tmp_path / "empty"

    with pytest.raises(DataValidationError, match="no aligned source partitions"):
        run_pipeline(TinySource(empty=True), output, expected_counts=None)

    assert not output.exists()


def test_run_baseline_mismatch_cleans_staging_output(tmp_path: Path) -> None:
    output = tmp_path / "mismatch"

    with pytest.raises(DataValidationError, match="pinned source baseline mismatch"):
        run_pipeline(TinySource(), output, expected_counts={"polygon_rows": 999})

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_run_records_expected_and_observed_baseline_counts(tmp_path: Path) -> None:
    result = run_pipeline(
        TinySource(),
        tmp_path / "baseline",
        expected_counts={"polygon_rows": 5, "yes": 2, "skipped_unsplit": 1},
    )

    baseline = result["baseline_verification"]
    assert baseline["checked"] is True
    assert baseline["expected_counts"] == {"polygon_rows": 5, "yes": 2, "skipped_unsplit": 1}
    assert baseline["observed_counts"]["yes"] == 2
    assert baseline["observed_counts"]["skipped_unsplit"] == 1


def test_run_writes_empty_overlap_when_no_yes_sentence_has_eunis(tmp_path: Path) -> None:
    source = TinySource()
    source.rows["labels"][0]["decision"] = "no"

    result = run_pipeline(source, tmp_path / "empty-overlap", expected_counts=None)

    assert result["stages"]["retained_overlap_rows"] == 0
    assert pq.read_table(tmp_path / "empty-overlap" / "overlap.parquet").num_rows == 0


def test_run_merges_multiple_shards_without_deduplicating_sentence_hashes(tmp_path: Path) -> None:
    result = run_pipeline(
        TinySource(multi_partition=True), tmp_path / "multi", expected_counts=None
    )

    stages = result["stages"]
    assert result["partition_count"] == 2
    assert stages["polygon_rows"] == 7
    assert stages["polygons_without_description"] == 2
    assert stages["description_tag_rows"] == 5
    assert stages["sentence_label_rows"] == 6
    assert stages["unique_sentence_occurrences"] == 6
    assert stages["unique_sentence_hashes"] == 4
    assert stages["retained_overlap_rows"] == 2
    assert result["retained_by_eunis_code"]["T11"] == 2
    assert pq.read_table(tmp_path / "multi" / "overlap.parquet").num_rows == 2


def test_pipeline_reads_only_required_projected_columns(tmp_path: Path) -> None:
    source = TinySource()
    # The fixture asserts its own projections remain the contract consumed by the pipeline.
    assert set(LABEL_COLUMNS) == set(source.rows["labels"][0])
    assert set(DESCRIPTION_COLUMNS) == set(source.rows["descriptions"][0])
    assert set(POLYGON_COLUMNS) == set(source.rows["polygons"][0])

    run_pipeline(source, tmp_path / "projected", expected_counts=None)
