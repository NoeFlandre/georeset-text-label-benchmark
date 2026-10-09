"""Freeze an audit-ready sample for the pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from georeset_text_label_benchmark.pilot.protocol import (
    CANDIDATE_LABELS_SHA256,
    OVERLAP_DATASET,
    OVERLAP_PARQUET_SHA256,
    OVERLAP_REVISION,
    SAMPLE_SEED,
    SAMPLE_SIZE,
)
from georeset_text_label_benchmark.pilot.sampling import (
    build_candidate_labels,
    select_distinct_sample,
)

SOURCE_COLUMNS = (
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
)
CANDIDATE_FIELDS = (
    "eunis_code",
    "eunis_name",
    "eunis_description",
    "candidate_text",
    "classification_release",
    "classification_source",
    "source_sha256",
    "license",
)
EXPECTED_SOURCE_ROWS = 224_789
EXPECTED_CANDIDATES = 158


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_staged_hashes(
    staging: Path,
    manifest: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    label: str,
) -> None:
    hashes = manifest.get("outputs_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"retained {label} staging manifest has no output hashes: {staging}")
    _validate_staged_hash_group(staging, hashes, input_paths, label, "input")
    _validate_staged_hash_group(staging, hashes, output_paths, label, "output")


def _validate_staged_hash_group(
    staging: Path,
    hashes: Mapping[str, Any],
    paths: Mapping[str, Path],
    label: str,
    kind: str,
) -> None:
    for name, path in paths.items():
        if hashes.get(name) != sha256_file(path):
            raise ValueError(f"retained {label} staging {kind} hash mismatch for {name}: {staging}")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _read_taxonomy(path: Path) -> dict[str, dict[str, str]]:
    taxonomy: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _validate_taxonomy_header(reader.fieldnames)
        for row in reader:
            code, details = _validated_taxonomy_row(row, taxonomy)
            taxonomy[code] = details
    return taxonomy


def _validate_taxonomy_header(fields: Sequence[str] | None) -> None:
    if tuple(fields or ()) != CANDIDATE_FIELDS:
        raise ValueError("candidate CSV schema does not match the pinned pilot schema")


def _validated_taxonomy_row(
    row: Mapping[str, str], existing: Mapping[str, Any]
) -> tuple[str, dict[str, str]]:
    code = row["eunis_code"]
    if not code or code in existing:
        raise ValueError(f"missing or duplicate candidate code: {code}")
    expected_text = f"{row['eunis_name']}\n{row['eunis_description']}"
    if row["candidate_text"] != expected_text:
        raise ValueError(f"candidate text mismatch for {code}")
    return code, {
        "name": row["eunis_name"],
        "description": row["eunis_description"],
        "classification_release": row["classification_release"],
        "classification_source": row["classification_source"],
        "source_sha256": row["source_sha256"],
        "license": row["license"],
    }


def _read_source_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    parquet = pq.ParquetFile(path)
    rows: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(columns=list(SOURCE_COLUMNS), batch_size=16_384):
        rows.extend(batch.to_pylist())
    return rows, parquet.metadata.num_rows


def _sample_metadata(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "row_count": len(rows),
        "unique_sentence_hash_count": len({row["text_sha256"] for row in rows}),
        "unique_polygon_count": len(
            {(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in rows}
        ),
        "observed_eunis_class_count": len({row["eunis_code"] for row in rows}),
    }


def _validate_freeze_inputs(
    source_parquet: Path,
    candidate_csv: Path,
    expected_source_sha256: str,
    expected_candidate_sha256: str | None,
    source_revision: str,
) -> tuple[str, str]:
    if source_revision != OVERLAP_REVISION:
        raise ValueError("source revision is not the pinned overlap snapshot")
    source_hash = sha256_file(source_parquet)
    if source_hash != expected_source_sha256:
        raise ValueError("overlap Parquet SHA-256 does not match the pinned input")
    candidate_hash = sha256_file(candidate_csv)
    if expected_candidate_sha256 is not None and candidate_hash != expected_candidate_sha256:
        raise ValueError("candidate CSV SHA-256 does not match the pinned input")
    return source_hash, candidate_hash


def _validate_source_row_count(parquet_row_count: int) -> None:
    if parquet_row_count != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"expected {EXPECTED_SOURCE_ROWS} overlap rows, got {parquet_row_count}")


def _validate_candidate_count(candidates: Sequence[Mapping[str, Any]]) -> None:
    if len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError(f"expected {EXPECTED_CANDIDATES} candidate classes")


def freeze_sample(
    source_parquet: Path,
    candidate_csv: Path,
    output_dir: Path,
    *,
    expected_source_sha256: str = OVERLAP_PARQUET_SHA256,
    expected_candidate_sha256: str | None = CANDIDATE_LABELS_SHA256,
    source_revision: str = OVERLAP_REVISION,
    size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """Write immutable selected rows and candidate text before predictions."""
    if output_dir.exists():
        raise FileExistsError(f"pilot directory already exists: {output_dir}")
    source_hash, candidate_hash = _validate_freeze_inputs(
        source_parquet,
        candidate_csv,
        expected_source_sha256,
        expected_candidate_sha256,
        source_revision,
    )
    source_rows, parquet_row_count = _read_source_rows(source_parquet)
    _validate_source_row_count(parquet_row_count)
    taxonomy = _read_taxonomy(candidate_csv)
    candidates = build_candidate_labels(source_rows, taxonomy)
    _validate_candidate_count(candidates)
    selected = select_distinct_sample(source_rows, size=size, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_output = output_dir / "candidate_labels.csv"
    shutil.copyfile(candidate_csv, candidate_output)
    sample = _frozen_sample(
        source_revision, source_hash, candidate_hash, source_rows, candidates, selected, size, seed
    )
    _write_json_exclusive(output_dir / "frozen_sample.json", sample)
    return _freeze_summary(output_dir, size, len(candidates), sample)


def _selection_metadata(
    selected: Sequence[Mapping[str, Any]], size: int, seed: int
) -> dict[str, Any]:
    ids = [row["sample_id"] for row in selected]
    return {
        "method": (
            "Sort occurrences by SHA256(seed + ':' + stable occurrence ID), then retain "
            "the first rows with unseen exact text hashes and unseen polygon keys."
        ),
        "seed": seed,
        "sample_size": size,
        "sample_ids_sha256": _sha256_json(ids),
        "unique_sentence_hash_count": len({row["text_sha256"] for row in selected}),
        "unique_polygon_count": len(
            {(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in selected}
        ),
    }


def _frozen_sample(
    source_revision: str,
    source_hash: str,
    candidate_hash: str,
    source_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    size: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "source": {
            "dataset": OVERLAP_DATASET,
            "revision": source_revision,
            "file": "overlap.parquet",
            "sha256": source_hash,
            "license": "OpenStreetMap ODbL-1.0; European Environment Agency CC-BY-4.0",
        },
        "source_coverage": _sample_metadata(source_rows),
        "selection": _selection_metadata(selected, size, seed),
        "candidate_labels": {
            "file": "candidate_labels.csv",
            "sha256": candidate_hash,
            "count": len(candidates),
            "codes": [row["eunis_code"] for row in candidates],
            "text_method": "Exact pinned EUNIS English name, newline, exact EEA description.",
        },
        "selected_rows": list(selected),
    }


def _freeze_summary(
    output_dir: Path, size: int, candidate_count: int, sample: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "sample_size": size,
        "candidate_count": candidate_count,
        "sample_ids_sha256": sample["selection"]["sample_ids_sha256"],
        "frozen_sample_sha256": sha256_file(output_dir / "frozen_sample.json"),
    }


def _validate_commit(commit: str, name: str) -> None:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"{name} must be a 40-character lowercase Git commit SHA")


def _validate_frozen_source(source: Mapping[str, Any]) -> None:
    if source["dataset"] != OVERLAP_DATASET or source["revision"] != OVERLAP_REVISION:
        raise ValueError("frozen sample uses a different overlap dataset revision")
    if source["sha256"] != OVERLAP_PARQUET_SHA256:
        raise ValueError("frozen sample uses a different overlap Parquet hash")


def _validate_frozen_rows(rows: list[dict[str, Any]], selection: Mapping[str, Any]) -> None:
    _validate_frozen_row_count(rows, selection)
    ids = [row["sample_id"] for row in rows]
    _require_unique_frozen_values(ids)
    _require_unique_frozen_values([row["text_sha256"] for row in rows])
    polygons = [(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in rows]
    _require_unique_frozen_values(polygons)
    _validate_frozen_ids(ids, selection)
    _validate_frozen_selection(rows, selection)


def _validate_frozen_row_count(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> None:
    if len(rows) != selection["sample_size"]:
        raise ValueError("frozen sample row count does not match its declaration")


def _require_unique_frozen_values(values: Sequence[Any]) -> None:
    if len(set(values)) != len(values):
        raise ValueError("frozen sample must have unique IDs, sentence hashes, and polygons")


def _validate_frozen_ids(ids: Sequence[str], selection: Mapping[str, Any]) -> None:
    if _sha256_json(ids) != selection["sample_ids_sha256"]:
        raise ValueError("frozen sample ID checksum mismatch")


def _validate_frozen_selection(rows: list[dict[str, Any]], selection: Mapping[str, Any]) -> None:
    expected = select_distinct_sample(rows, size=selection["sample_size"], seed=selection["seed"])
    if expected != rows:
        raise ValueError("frozen rows do not match the deterministic sampling protocol")


def _read_frozen_sample(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen_path = run_dir / "frozen_sample.json"
    with frozen_path.open(encoding="utf-8") as stream:
        sample = json.load(stream)
    _validate_frozen_source(sample["source"])
    rows = sample["selected_rows"]
    _validate_frozen_rows(rows, sample["selection"])
    return sample, rows


def _candidate_rows(run_dir: Path, sample: Mapping[str, Any]) -> list[dict[str, str]]:
    candidate_path = run_dir / sample["candidate_labels"]["file"]
    if sha256_file(candidate_path) != sample["candidate_labels"]["sha256"]:
        raise ValueError("candidate label file checksum mismatch")
    taxonomy = _read_taxonomy(candidate_path)
    candidates = _candidate_records(taxonomy)
    _validate_candidate_references(candidates, sample)
    return candidates


def read_frozen_pilot_inputs(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    """Read and validate the existing frozen sample and candidate vocabulary."""
    sample, rows = _read_frozen_sample(run_dir)
    candidates = _candidate_rows(run_dir, sample)
    return sample, rows, candidates


def _candidate_records(taxonomy: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    candidates = [
        {
            "eunis_code": code,
            "eunis_name": details["name"],
            "candidate_text": f"{details['name']}\n{details['description']}",
            "classification_release": details["classification_release"],
            "classification_source": details["classification_source"],
            "source_sha256": details["source_sha256"],
            "license": details["license"],
        }
        for code, details in sorted(taxonomy.items())
    ]
    _validate_candidate_count(candidates)
    return candidates


def _validate_candidate_references(
    candidates: Sequence[Mapping[str, str]], sample: Mapping[str, Any]
) -> None:
    _validate_candidate_code_list(candidates, sample)
    names = {row["eunis_code"]: row["eunis_name"] for row in candidates}
    _validate_sampled_candidate_names(sample, names)


def _validate_candidate_code_list(
    candidates: Sequence[Mapping[str, str]], sample: Mapping[str, Any]
) -> None:
    if [row["eunis_code"] for row in candidates] != sample["candidate_labels"]["codes"]:
        raise ValueError("candidate code list does not match the frozen protocol")


def _validate_sampled_candidate_names(sample: Mapping[str, Any], names: Mapping[str, str]) -> None:
    for row in sample["selected_rows"]:
        if names.get(row["eunis_code"]) != row["eunis_name"]:
            raise ValueError(f"frozen gold name mismatch for {row['eunis_code']}")


def _candidate_provenance(
    sample: Mapping[str, Any], candidates: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    """Share the same pinned EEA attribution in both pilot run manifests."""
    releases = sorted({row["classification_release"] for row in candidates})
    citations = sorted({row["classification_source"] for row in candidates})
    taxonomy_hashes = sorted({row["source_sha256"] for row in candidates})
    return {
        **sample["candidate_labels"],
        "classification_releases": releases,
        "classification_sources": citations,
        "taxonomy_archive_sha256": taxonomy_hashes,
        "license": "CC-BY-4.0, European Environment Agency",
    }
