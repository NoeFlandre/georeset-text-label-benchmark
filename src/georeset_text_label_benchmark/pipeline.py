"""Bounded, reproducible execution and reporting for the overlap join."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import traceback
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
from pyarrow import parquet as pq

from georeset_text_label_benchmark.config import (
    EXPECTED_SOURCE_COUNTS,
    DatasetSnapshots,
)
from georeset_text_label_benchmark.errors import DataValidationError
from georeset_text_label_benchmark.join import process_partition
from georeset_text_label_benchmark.models import AuditCounts, GlobalKeys, SourcePartition
from georeset_text_label_benchmark.output_schema import LABELLED_SCHEMA, OVERLAP_SCHEMA
from georeset_text_label_benchmark.source import (
    DESCRIPTION_COLUMNS,
    LABEL_COLUMNS,
    POLYGON_COLUMNS,
)

COMMIT_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
HASH_CHUNK_BYTES = 1024 * 1024


class OverlapSource(Protocol):
    """Narrow read-only interface used by the pipeline and its local tests."""

    snapshots: DatasetSnapshots

    def partitions(self) -> list[SourcePartition]: ...

    def read_rows(self, path: str, columns: Sequence[str]) -> Iterable[dict[str, Any]]: ...

    def eunis_manifest(self) -> dict[str, Any]: ...


def run_pipeline(
    source: OverlapSource,
    output_dir: str | Path,
    *,
    computation_commit: str,
    validation_commit: str,
    expected_counts: Mapping[str, int] | None = EXPECTED_SOURCE_COUNTS,
) -> dict[str, Any]:
    """Join aligned shards, write only retained occurrences, and atomically publish artifacts."""
    _validate_commit("computation_commit", computation_commit)
    _validate_commit("validation_commit", validation_commit)
    destination = Path(output_dir)
    if destination.exists():
        raise DataValidationError(f"output path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        summary = _build_run(source, stage, expected_counts, computation_commit, validation_commit)
        os.replace(stage, destination)
    except Exception as error:
        try:
            _preserve_failure_diagnostic(destination, stage, error)
        except OSError as diagnostic_error:
            error.add_note(
                f"Could not write failure diagnostics ({diagnostic_error}); "
                f"staging directory retained at {stage}"
            )
            raise error.with_traceback(error.__traceback__) from None
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return summary


def _validate_commit(field: str, value: str) -> None:
    if not isinstance(value, str) or COMMIT_SHA_PATTERN.fullmatch(value) is None:
        raise DataValidationError(f"{field} must be a 40-character hexadecimal Git commit SHA")


def _preserve_failure_diagnostic(destination: Path, stage: Path, error: Exception) -> None:
    record = {
        "status": "failed",
        "output_path": str(destination),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "staged_files": _staged_file_inventory(stage),
    }
    descriptor, _path = tempfile.mkstemp(
        prefix=f".{destination.name}-failure-", suffix=".json", dir=destination.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _staged_file_inventory(stage: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(stage).as_posix(), "size_bytes": path.stat().st_size}
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    ]


def _build_run(
    source: OverlapSource,
    stage: Path,
    expected_counts: Mapping[str, int] | None,
    computation_commit: str,
    validation_commit: str,
) -> dict[str, Any]:
    partitions = source.partitions()
    if not partitions:
        raise DataValidationError("no aligned source partitions were discovered")
    manifest = source.eunis_manifest()
    audit = AuditCounts()
    keys = GlobalKeys()
    output_path = stage / "overlap.parquet"
    labelled_path = stage / "labelled-eunis.parquet"
    writer = pq.ParquetWriter(output_path, OVERLAP_SCHEMA, compression="zstd")
    labelled_writer = pq.ParquetWriter(labelled_path, LABELLED_SCHEMA, compression="zstd")
    try:
        for partition in partitions:
            result = _process_source_partition(
                source,
                partition,
                keys,
                source.snapshots.input_revision,
                manifest["eunis_reference_version"],
            )
            _write_partition(writer, labelled_writer, result.overlap_rows)
            audit.merge(result.audit)
    finally:
        writer.close()
        labelled_writer.close()
    summary = _summary(audit, keys, partitions, manifest, expected_counts)
    summary_path = stage / "summary.json"
    _write_json(summary_path, summary)
    artifact_sha256 = {
        "overlap.parquet": _sha256_file(output_path),
        "labelled-eunis.parquet": _sha256_file(labelled_path),
        "summary.json": _sha256_file(summary_path),
    }
    _write_json(
        stage / "manifest.json",
        _manifest(
            source.snapshots,
            manifest,
            summary,
            artifact_sha256,
            computation_commit,
            validation_commit,
        ),
    )
    return summary


def _write_partition(
    writer: pq.ParquetWriter,
    labelled_writer: pq.ParquetWriter,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    yes_rows = [_overlap_only(row) for row in rows if row["decision"] == "yes"]
    if yes_rows:
        writer.write_table(pa.Table.from_pylist(yes_rows, schema=OVERLAP_SCHEMA))
    if rows:
        labelled_writer.write_table(pa.Table.from_pylist(list(rows), schema=LABELLED_SCHEMA))


def _overlap_only(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field.name: row[field.name] for field in OVERLAP_SCHEMA}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        chunk = stream.read(HASH_CHUNK_BYTES)
        while chunk:
            digest.update(chunk)
            chunk = stream.read(HASH_CHUNK_BYTES)
    return digest.hexdigest()


def _process_source_partition(
    source: OverlapSource,
    partition: SourcePartition,
    keys: GlobalKeys,
    input_revision: str,
    expected_eunis_source_version: str,
):
    labels = source.read_rows(partition.labels_path, LABEL_COLUMNS)
    descriptions = source.read_rows(partition.descriptions_path, DESCRIPTION_COLUMNS)
    polygons = source.read_rows(partition.polygons_path, POLYGON_COLUMNS)
    return process_partition(
        labels,
        descriptions,
        polygons,
        input_revision=input_revision,
        partition_name=partition.name,
        expected_eunis_source_version=expected_eunis_source_version,
        global_keys=keys,
        labelled=True,
    )


def _summary(
    audit: AuditCounts,
    keys: GlobalKeys,
    partitions: list[SourcePartition],
    eunis_manifest: Mapping[str, Any],
    expected_counts: Mapping[str, int] | None,
) -> dict[str, Any]:
    stages = {
        "polygon_rows": audit.polygon_rows,
        "assigned_polygon_rows": audit.assigned_polygon_rows,
        "polygons_without_description": len(keys.polygon_keys - keys.described_polygon_keys),
        "description_tag_rows": audit.description_tag_rows,
        "sentence_label_rows": audit.label_rows,
        "sentence_rows": audit.sentence_rows,
        "yes_sentence_rows": audit.decisions["yes"],
        "no_sentence_rows": audit.decisions["no"],
        "failed_sentence_rows": audit.decisions["failed"],
        "skipped_unsplit_rows": audit.skipped_unsplit_rows,
        "yes_without_eunis_rows": audit.yes_without_eunis_rows,
        "retained_overlap_rows": audit.retained_rows,
        "unique_polygon_keys": len(keys.polygon_keys),
        "unique_description_tag_keys": len(keys.description_keys),
        "unique_sentence_occurrences": len(keys.sentence_keys),
        "unique_sentence_hashes": len(audit.unique_sentence_hashes),
    }
    _verify_expected_counts(stages, audit, expected_counts)
    return {
        "status": "complete",
        "partition_count": len(partitions),
        "eunis": dict(eunis_manifest),
        "stages": stages,
        "labels_by_decision_and_eunis_assignment": _assignment_summary(audit),
        "retained_by_eunis_code": _counter_dict(audit.overlap_by_eunis),
        "retained_by_tag_key": _counter_dict(audit.overlap_by_tag),
        "retained_by_language_code": _counter_dict(audit.overlap_by_language),
        "baseline_verification": _baseline_result(stages, audit, expected_counts),
    }


def _verify_expected_counts(
    stages: Mapping[str, int], audit: AuditCounts, expected: Mapping[str, int] | None
) -> None:
    if expected is None:
        return
    actual = _observed_counts(stages, audit)
    mismatches = [
        f"{name}: expected {value}, observed {actual.get(name)}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise DataValidationError("pinned source baseline mismatch: " + "; ".join(mismatches))


def _observed_counts(stages: Mapping[str, int], audit: AuditCounts) -> dict[str, int]:
    return {
        **stages,
        "label_rows": audit.label_rows,
        "yes": audit.decisions["yes"],
        "no": audit.decisions["no"],
        "failed": audit.decisions["failed"],
        "skipped_unsplit": audit.decisions["skipped_unsplit"],
    }


def _baseline_result(
    stages: Mapping[str, int], audit: AuditCounts, expected: Mapping[str, int] | None
) -> dict[str, Any]:
    if expected is None:
        return {"checked": False, "expected_counts": None}
    return {
        "checked": True,
        "expected_counts": dict(expected),
        "observed_counts": _observed_counts(stages, audit),
    }


def _assignment_summary(audit: AuditCounts) -> dict[str, dict[str, int]]:
    decisions = sorted(audit.decisions)
    assignments = ("assigned", "missing")
    return {
        decision: {
            assignment: audit.decision_assignment[(decision, assignment)]
            for assignment in assignments
        }
        for decision in decisions
    }


def _counter_dict(counter: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(
    snapshots: DatasetSnapshots,
    eunis_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifact_sha256: Mapping[str, str],
    computation_commit: str,
    validation_commit: str,
) -> dict[str, Any]:
    return {
        "project": "georeset-text-label-benchmark",
        "project_version": "0.1.0",
        "author": "Noé Flandre",
        "computation_commit": computation_commit,
        "validation_commit": validation_commit,
        "method": "exact sentence-occurrence join; no geometry recomputation",
        "join_keys": {
            "label_to_description": ["description_identity", "tag_key"],
            "sentence_occurrence": ["description_identity", "tag_key", "sentence_index"],
            "description_to_polygon": ["osm_type", "osm_id"],
            "additional_provenance_check": "source_pbf equality",
        },
        "input_snapshots": {
            "labels_repo": snapshots.labels_repo,
            "labels_revision": snapshots.labels_revision,
            "eunis_repo": snapshots.eunis_repo,
            "eunis_revision": snapshots.eunis_revision,
            "shared_input_revision": snapshots.input_revision,
            **dict(eunis_manifest),
        },
        "output_schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in OVERLAP_SCHEMA
        ],
        "labelled_output_schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in LABELLED_SCHEMA
        ],
        "output_row_count": summary["stages"]["retained_overlap_rows"],
        "artifact_sha256": dict(artifact_sha256),
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
