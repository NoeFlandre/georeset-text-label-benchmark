"""Small local end-to-end checks for artifact publication and audit counts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pyarrow import parquet as pq

from georeset_text_label_benchmark import pipeline as pipeline_module
from georeset_text_label_benchmark.config import EXPECTED_SOURCE_COUNTS, DatasetSnapshots
from georeset_text_label_benchmark.errors import DataValidationError, ProvenanceError
from georeset_text_label_benchmark.models import SourcePartition
from georeset_text_label_benchmark.output_schema import LABELLED_SCHEMA
from georeset_text_label_benchmark.pipeline import (
    DESCRIPTION_COLUMNS,
    LABEL_COLUMNS,
    POLYGON_COLUMNS,
    _sha256_file,
    _staged_file_inventory,
    _write_json,
    run_pipeline,
)

REVISION = "shared-revision"
PBF = "tuvalu-latest.osm.pbf"
SNAPSHOTS = DatasetSnapshots("owner/labels", "label-sha", "owner/eunis", "eunis-sha", REVISION)
TINY_BASELINE_COUNTS = {
    "polygon_rows": 5,
    "assigned_polygon_rows": 3,
    "description_tag_rows": 4,
    "label_rows": 5,
    "sentence_rows": 4,
    "yes": 2,
    "no": 1,
    "failed": 1,
    "skipped_unsplit": 1,
}
TEST_COMMIT = "a" * 40


def _run_pipeline(
    source: TinySource,
    output_dir: Path,
    *,
    expected_counts: dict[str, int] | None = EXPECTED_SOURCE_COUNTS,
    computation_commit: str = TEST_COMMIT,
    validation_commit: str = TEST_COMMIT,
) -> dict[str, Any]:
    return run_pipeline(
        source,
        output_dir,
        expected_counts=expected_counts,
        computation_commit=computation_commit,
        validation_commit=validation_commit,
    )


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
                    "labels-2": [
                        _label("d5", "description", 0, repeated_text, "yes"),
                        _label(
                            "d6",
                            "description:en",
                            0,
                            "Whole unsplit value.",
                            "skipped_unsplit",
                            language="tgl",
                        ),
                    ],
                    "descriptions-2": [
                        _description("d5", "description", 6, [repeated_text]),
                        _description("d6", "description:en", 8, [], language="tgl"),
                    ],
                    "polygons-2": [_polygon(6, "T11"), _polygon(7, "T44"), _polygon(8, None)],
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
        "bbox_min_x": 10.0 + osm_id,
        "bbox_min_y": 20.0 + osm_id,
        "bbox_max_x": 10.5 + osm_id,
        "bbox_max_y": 20.5 + osm_id,
    }


def test_pipeline_stage_prefix_identifies_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "published-run"
    original_mkdtemp = pipeline_module.tempfile.mkdtemp
    prefixes: list[str | None] = []

    def record_mkdtemp(*, prefix: str | None = None, dir: str | Path | None = None) -> str:
        prefixes.append(prefix)
        return original_mkdtemp(prefix=prefix, dir=dir)

    def stop_after_stage(*_args: Any) -> dict[str, Any]:
        raise RuntimeError("stop after staging directory allocation")

    def fail_diagnostic(*_args: Any) -> None:
        raise OSError("leave stage for the test")

    monkeypatch.setattr(pipeline_module.tempfile, "mkdtemp", record_mkdtemp)
    monkeypatch.setattr(pipeline_module, "_build_run", stop_after_stage)
    monkeypatch.setattr(pipeline_module, "_preserve_failure_diagnostic", fail_diagnostic)

    with pytest.raises(RuntimeError, match="stop after staging directory allocation"):
        _run_pipeline(TinySource(), output, expected_counts=None)

    assert prefixes == [f".{output.name}-"]


def test_run_writes_projected_occurrences_and_stage_report(tmp_path: Path) -> None:
    output = tmp_path / "run"

    result = _run_pipeline(TinySource(), output, expected_counts=None)

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
    parquet_metadata = pq.ParquetFile(output / "overlap.parquet").metadata
    assert parquet_metadata.num_row_groups == 1
    assert parquet_metadata.row_group(0).column(0).compression == "ZSTD"
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
    assert result["baseline_verification"] == {"checked": False, "expected_counts": None}
    assert json.loads((output / "manifest.json").read_text())["output_row_count"] == 1
    assert json.loads((output / "summary.json").read_text()) == result


def test_run_refuses_existing_output_without_touching_it(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(DataValidationError, match="already exists"):
        _run_pipeline(TinySource(), output, expected_counts=None)

    assert sentinel.read_text() == "keep"


def test_run_rejects_empty_source_and_leaves_no_published_output(tmp_path: Path) -> None:
    output = tmp_path / "empty"

    with pytest.raises(
        DataValidationError,
        match=r"no aligned source partitions were discovered",
    ) as caught:
        _run_pipeline(TinySource(empty=True), output, expected_counts=None)

    assert str(caught.value) == "no aligned source partitions were discovered"
    assert not output.exists()


def test_run_baseline_mismatch_cleans_staging_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "mismatch"
    remove_tree = pipeline_module.shutil.rmtree
    removals: list[tuple[Path, dict[str, Any]]] = []

    def record_removal(path: Path, *, ignore_errors: bool = False) -> None:
        kwargs = {"ignore_errors": ignore_errors}
        removals.append((path, kwargs))
        remove_tree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(pipeline_module.shutil, "rmtree", record_removal)

    with pytest.raises(DataValidationError, match="pinned source baseline mismatch"):
        _run_pipeline(TinySource(), output, expected_counts={"polygon_rows": 999})

    assert not output.exists()
    diagnostics = list(tmp_path.glob(".mismatch-failure-*.json"))
    assert len(diagnostics) == 1
    diagnostic = json.loads(diagnostics[0].read_text())
    assert set(diagnostic) == {
        "status",
        "output_path",
        "error_type",
        "error_message",
        "traceback",
        "staged_files",
    }
    assert diagnostic["status"] == "failed"
    assert diagnostic["output_path"] == str(output)
    assert diagnostic["error_type"] == "DataValidationError"
    assert diagnostic["error_message"] == (
        "pinned source baseline mismatch: polygon_rows: expected 999, observed 5"
    )
    assert "_verify_expected_counts" in diagnostic["traceback"]
    staged = {entry["path"]: entry["size_bytes"] for entry in diagnostic["staged_files"]}
    assert set(staged) == {"labelled-eunis.parquet", "overlap.parquet"}
    assert all(size > 0 for size in staged.values())
    assert diagnostics[0].read_text(encoding="utf-8") == (
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    )
    assert len(removals) == 1
    assert removals[0][1] == {"ignore_errors": True}
    assert list(tmp_path.glob(".mismatch-*")) == diagnostics


def test_run_preserves_stage_when_failure_diagnostic_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_to_write_diagnostic(*_args: Any) -> None:
        raise OSError("diagnostic storage unavailable")

    monkeypatch.setattr(pipeline_module, "_preserve_failure_diagnostic", fail_to_write_diagnostic)
    with pytest.raises(DataValidationError, match="pinned source baseline mismatch") as caught:
        _run_pipeline(TinySource(), tmp_path / "diagnostic-failure", expected_counts={"yes": 999})

    assert "diagnostic storage unavailable" in caught.value.__notes__[0]
    stages = [path for path in tmp_path.glob(".diagnostic-failure-*") if path.is_dir()]
    assert len(stages) == 1
    assert (stages[0] / "overlap.parquet").is_file()
    assert "_verify_expected_counts" in "".join(traceback.format_exception(caught.value))


def test_failure_diagnostic_is_sorted_utf8_and_preserves_original_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "partial.parquet").write_bytes(b"partial")
    output = tmp_path / "output"

    def raise_value_error() -> None:
        raise ValueError("café")

    try:
        raise_value_error()
    except ValueError as caught_error:
        error = caught_error
    open_file = os.fdopen
    descriptors: list[tuple[str, str | None]] = []

    def record_open(
        descriptor: int,
        mode: str = "r",
        *args: Any,
        encoding: str | None = None,
        **kwargs: Any,
    ) -> Any:
        descriptors.append((mode, encoding))
        return open_file(descriptor, mode, *args, encoding=encoding, **kwargs)

    monkeypatch.setattr(pipeline_module.os, "fdopen", record_open)
    pipeline_module._preserve_failure_diagnostic(output, stage, error)

    diagnostic_path = next(tmp_path.glob(".output-failure-*.json"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert descriptors == [("w", "utf-8")]
    assert diagnostic == {
        "status": "failed",
        "output_path": str(output),
        "error_type": "ValueError",
        "error_message": "café",
        "traceback": "".join(traceback.format_exception(error)),
        "staged_files": [{"path": "partial.parquet", "size_bytes": 7}],
    }
    assert diagnostic_path.read_bytes() == (
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_run_rejects_mismatched_label_row_baseline(tmp_path: Path) -> None:
    with pytest.raises(
        DataValidationError,
        match="label_rows: expected 6, observed 5",
    ):
        _run_pipeline(TinySource(), tmp_path / "label-mismatch", expected_counts={"label_rows": 6})

    assert not (tmp_path / "label-mismatch").exists()


def test_run_reports_all_baseline_mismatches_in_sorted_order(tmp_path: Path) -> None:
    with pytest.raises(
        DataValidationError,
        match=re.escape(
            "pinned source baseline mismatch: yes: expected 3, observed 2; "
            "polygon_rows: expected 0, observed 5"
        ),
    ):
        _run_pipeline(
            TinySource(),
            tmp_path / "multi-mismatch",
            expected_counts={"yes": 3, "polygon_rows": 0},
        )


@pytest.mark.parametrize(("name", "observed"), list(TINY_BASELINE_COUNTS.items()))
def test_run_rejects_mismatch_for_every_baseline_field(
    tmp_path: Path, name: str, observed: int
) -> None:
    with pytest.raises(
        DataValidationError,
        match=f"{name}: expected {observed + 1}, observed {observed}",
    ):
        _run_pipeline(
            TinySource(), tmp_path / "count-mismatch", expected_counts={name: observed + 1}
        )

    assert not (tmp_path / "count-mismatch").exists()
    diagnostics = list(tmp_path.glob(".count-mismatch-failure-*.json"))
    assert len(diagnostics) == 1
    assert json.loads(diagnostics[0].read_text())["error_message"].endswith(
        f"{name}: expected {observed + 1}, observed {observed}"
    )


def test_run_records_expected_and_observed_baseline_counts(tmp_path: Path) -> None:
    expected = TINY_BASELINE_COUNTS
    assert set(expected) == set(EXPECTED_SOURCE_COUNTS)
    output = tmp_path / "baseline"
    result = _run_pipeline(
        TinySource(),
        output,
        expected_counts=expected,
    )

    baseline = result["baseline_verification"]
    assert baseline["checked"] is True
    assert baseline["expected_counts"] == expected
    assert {name: baseline["observed_counts"][name] for name in expected} == expected
    assert result == {
        "status": "complete",
        "partition_count": 1,
        "eunis": TinySource().eunis_manifest(),
        "stages": {
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
        },
        "labels_by_decision_and_eunis_assignment": {
            "failed": {"assigned": 1, "missing": 0},
            "no": {"assigned": 1, "missing": 0},
            "skipped_unsplit": {"assigned": 0, "missing": 1},
            "yes": {"assigned": 1, "missing": 1},
        },
        "retained_by_eunis_code": {"T11": 1},
        "retained_by_tag_key": {"description": 1},
        "retained_by_language_code": {"eng": 1},
        "baseline_verification": {
            "checked": True,
            "expected_counts": TINY_BASELINE_COUNTS,
            "observed_counts": {
                **TINY_BASELINE_COUNTS,
                "polygons_without_description": 1,
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
                "sentence_label_rows": 5,
            },
        },
    }
    assert json.loads((output / "summary.json").read_text()) == result
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["output_row_count"] == 1
    assert manifest["join_keys"]["label_to_description"] == [
        "description_identity",
        "tag_key",
    ]
    expected_schema = [
        ("source_pbf", "string", False),
        ("osm_type", "string", False),
        ("osm_id", "int64", False),
        ("description_identity", "string", False),
        ("tag_key", "string", False),
        ("sentence_index", "int32", False),
        ("sentence", "string", False),
        ("text_sha256", "string", False),
        ("language_code", "string", True),
        ("eunis_code", "string", False),
        ("eunis_name", "string", False),
        ("eunis_overlap_percentage", "double", False),
        ("eunis_source_version", "string", False),
    ]
    assert manifest == {
        "project": "georeset-text-label-benchmark",
        "project_version": "0.1.0",
        "author": "Noé Flandre",
        "computation_commit": TEST_COMMIT,
        "validation_commit": TEST_COMMIT,
        "method": "exact sentence-occurrence join; no geometry recomputation",
        "join_keys": {
            "label_to_description": ["description_identity", "tag_key"],
            "sentence_occurrence": ["description_identity", "tag_key", "sentence_index"],
            "description_to_polygon": ["osm_type", "osm_id"],
            "additional_provenance_check": "source_pbf equality",
        },
        "input_snapshots": {
            "labels_repo": "owner/labels",
            "labels_revision": "label-sha",
            "eunis_repo": "owner/eunis",
            "eunis_revision": "eunis-sha",
            "shared_input_revision": REVISION,
            **TinySource().eunis_manifest(),
        },
        "output_schema": [
            {"name": name, "type": data_type, "nullable": nullable}
            for name, data_type, nullable in expected_schema
        ],
        "output_row_count": 1,
        "labelled_output_schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in LABELLED_SCHEMA
        ],
        "artifact_sha256": {
            "overlap.parquet": hashlib.sha256(
                (output / "overlap.parquet").read_bytes()
            ).hexdigest(),
            "labelled-eunis.parquet": hashlib.sha256(
                (output / "labelled-eunis.parquet").read_bytes()
            ).hexdigest(),
            "summary.json": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
        },
        "eunis_context_is_polygon_context_not_sentence_ground_truth": True,
        "license_and_attribution": {
            "project_code": "Apache-2.0",
            "osm_source_data": "OpenStreetMap ODbL 1.0; attribution required",
            "eunis_reference_data": "EEA EUNIS probability maps v1 (2021), CC-BY 4.0",
            "eea_legal_notice": "https://www.eea.europa.eu/en/legal-notice",
            "eea_collection_factsheets": [
                "https://sdi.eea.europa.eu/catalogue/datahub/api/records/99498d2c-7350-4655-b914-92c5b9e016c5/formatters/xsl-view?approved=true&language=eng&output=pdf",
                "https://sdi.eea.europa.eu/catalogue/datahub/api/records/f425a73e-dc6c-40be-93d6-9d234d4bbe1b/formatters/xsl-view?approved=true&language=eng&output=pdf",
                "https://sdi.eea.europa.eu/catalogue/datahub/api/records/443cdba1-1d4b-4cae-91d5-72fa99a8a758/formatters/xsl-view?approved=true&language=eng&output=pdf",
                "https://sdi.eea.europa.eu/catalogue/datahub/api/records/0c4270c4-fd7e-4099-ac2f-5af2079ddbd8/formatters/xsl-view?approved=true&language=eng&output=pdf",
                "https://sdi.eea.europa.eu/catalogue/datahub/api/records/ae2fdada-93d6-4cf1-a11c-6ebccf25d286/formatters/xsl-view?approved=true&language=eng&output=pdf",
                "https://sdi.eea.europa.eu/catalogue/datahub/api/records/7a2d78ec-8a31-4e7b-91df-45db6e64e842/formatters/xsl-view?approved=true&language=eng&output=pdf",
                "https://sdi.eea.europa.eu/catalogue/datahub/api/records/a6c48c2d-114f-406e-ba99-c9085a5f5aee/formatters/xsl-view?approved=true&language=eng&output=pdf",
            ],
        },
    }


def test_run_creates_nested_directory_and_uses_canonical_artifact_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "new-parent" / "nested" / "run"
    write_json = pipeline_module._write_json
    names: list[str] = []

    def record_write(path: Path, value: dict[str, Any]) -> None:
        names.append(path.name)
        write_json(path, value)

    monkeypatch.setattr(pipeline_module, "_write_json", record_write)
    _run_pipeline(TinySource(), output, expected_counts=None)

    assert names == ["summary.json", "manifest.json"]
    assert {path.name for path in output.iterdir()} == {
        "overlap.parquet",
        "labelled-eunis.parquet",
        "summary.json",
        "manifest.json",
    }


def test_run_rejects_eunis_version_that_differs_from_manifest(tmp_path: Path) -> None:
    source = TinySource()
    source.rows["polygons"][0]["eunis_source_version"] = "different-reference-v999"

    with pytest.raises(ProvenanceError, match="pinned EUNIS reference version"):
        _run_pipeline(source, tmp_path / "eunis-version-mismatch", expected_counts=None)

    assert not (tmp_path / "eunis-version-mismatch").exists()
    assert len(list(tmp_path.glob(".eunis-version-mismatch-failure-*.json"))) == 1


def test_run_writes_empty_overlap_when_no_yes_sentence_has_eunis(tmp_path: Path) -> None:
    source = TinySource()
    source.rows["labels"][0]["decision"] = "no"

    result = _run_pipeline(source, tmp_path / "empty-overlap", expected_counts=None)

    assert result["stages"]["retained_overlap_rows"] == 0
    assert pq.read_table(tmp_path / "empty-overlap" / "overlap.parquet").num_rows == 0


def test_run_merges_multiple_shards_without_deduplicating_sentence_hashes(tmp_path: Path) -> None:
    result = _run_pipeline(
        TinySource(multi_partition=True), tmp_path / "multi", expected_counts=None
    )

    stages = result["stages"]
    assert result["partition_count"] == 2
    assert stages["polygon_rows"] == 8
    assert stages["assigned_polygon_rows"] == 5
    assert stages["polygons_without_description"] == 2
    assert stages["description_tag_rows"] == 6
    assert stages["sentence_label_rows"] == 7
    assert stages["sentence_rows"] == 5
    assert stages["skipped_unsplit_rows"] == 2
    assert stages["yes_without_eunis_rows"] == 1
    assert stages["unique_sentence_occurrences"] == 7
    assert stages["unique_sentence_hashes"] == 4
    assert stages["retained_overlap_rows"] == 2
    assert result["retained_by_eunis_code"]["T11"] == 2
    assert result["labels_by_decision_and_eunis_assignment"]["skipped_unsplit"] == {
        "assigned": 0,
        "missing": 2,
    }
    assert pq.read_table(tmp_path / "multi" / "overlap.parquet").num_rows == 2


def test_pipeline_errors_keep_source_partition_context(tmp_path: Path) -> None:
    source = TinySource()
    source.rows["labels"][0]["decision"] = None

    with pytest.raises(
        DataValidationError,
        match=re.escape("tuvalu-latest.parquet label: decision must be a non-empty string"),
    ) as caught:
        _run_pipeline(source, tmp_path / "invalid-row", expected_counts=None)

    assert str(caught.value) == "tuvalu-latest.parquet label: decision must be a non-empty string"


def test_pipeline_reads_only_required_projected_columns(tmp_path: Path) -> None:
    source = TinySource()
    # The fixture asserts its own projections remain the contract consumed by the pipeline.
    assert set(LABEL_COLUMNS) == set(source.rows["labels"][0])
    assert set(DESCRIPTION_COLUMNS) == set(source.rows["descriptions"][0])
    assert set(POLYGON_COLUMNS) == set(source.rows["polygons"][0])

    _run_pipeline(source, tmp_path / "projected", expected_counts=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("computation_commit", "main"),
        ("validation_commit", "z" * 40),
        ("computation_commit", None),
    ],
)
def test_run_rejects_missing_or_malformed_git_commit_provenance(
    tmp_path: Path, field: str, value: Any
) -> None:
    commits: dict[str, Any] = {
        "computation_commit": TEST_COMMIT,
        "validation_commit": TEST_COMMIT,
    }
    commits[field] = value
    with pytest.raises(DataValidationError, match=f"{field} must be a 40-character"):
        run_pipeline(
            TinySource(),
            tmp_path / "invalid-commit",
            **commits,
            expected_counts=None,
        )

    assert not (tmp_path / "invalid-commit").exists()


def test_sha256_hashes_files_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"a" * (1024 * 1024 + 37)
    path = tmp_path / "large.bin"
    read_sizes: list[int | None] = []

    class RecordingStream(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            read_sizes.append(size)
            return super().read(size)

    def recording_open(opened_path: Path, mode: str = "r") -> RecordingStream:
        assert opened_path == path
        assert mode == "rb"
        return RecordingStream(payload)

    monkeypatch.setattr(Path, "open", recording_open)
    assert _sha256_file(path) == hashlib.sha256(payload).hexdigest()
    assert read_sizes == [1024 * 1024, 1024 * 1024, 1024 * 1024]


def test_staged_file_inventory_is_sorted_and_relative(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "nested").mkdir(parents=True)
    (stage / "zeta.txt").write_bytes(b"123")
    (stage / "nested" / "alpha.txt").write_bytes(b"12")

    assert _staged_file_inventory(stage) == [
        {"path": "nested/alpha.txt", "size_bytes": 2},
        {"path": "zeta.txt", "size_bytes": 3},
    ]


def test_write_json_uses_sorted_utf8_and_one_trailing_newline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_text = Path.write_text
    observed_encoding: list[str | None] = []

    def record_encoding(
        path: Path, value: str, *, encoding: str | None = None, newline: str | None = None
    ) -> int:
        observed_encoding.append(encoding)
        return write_text(path, value, encoding=encoding, newline=newline)

    monkeypatch.setattr(Path, "write_text", record_encoding)
    output = tmp_path / "json.json"
    _write_json(output, {"é": 2, "z": 1})

    assert observed_encoding == ["utf-8"]
    assert output.read_bytes() == b'{\n  "z": 1,\n  "\\u00e9": 2\n}\n'


def test_run_writes_labelled_pool_with_yes_and_no_rows_beside_yes_only_overlap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _run_pipeline(TinySource(), output, expected_counts=TINY_BASELINE_COUNTS)

    labelled = pq.read_table(output / "labelled-eunis.parquet").to_pylist()
    overlap = pq.read_table(output / "overlap.parquet")

    assert sorted((row["decision"], row["osm_id"]) for row in labelled) == [
        ("no", 1),
        ("yes", 1),
    ]
    assert (labelled[0]["bbox_min_x"], labelled[0]["bbox_min_y"]) == (11.0, 21.0)
    assert "decision" not in overlap.column_names
    assert overlap.num_rows == 1


def test_labelled_pool_is_listed_in_the_manifest_with_its_checksum(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _run_pipeline(TinySource(), output, expected_counts=TINY_BASELINE_COUNTS)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checksums = manifest["artifact_sha256"]
    assert checksums["labelled-eunis.parquet"] == _sha256_file(output / "labelled-eunis.parquet")


def test_polygon_projection_includes_the_bbox_columns() -> None:
    assert {"bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y"} <= set(POLYGON_COLUMNS)


def test_both_pipeline_outputs_are_zstd_compressed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _run_pipeline(TinySource(), output, expected_counts=TINY_BASELINE_COUNTS)

    for name in ("overlap.parquet", "labelled-eunis.parquet"):
        column = pq.ParquetFile(output / name).metadata.row_group(0).column(0)
        assert column.compression == "ZSTD"
