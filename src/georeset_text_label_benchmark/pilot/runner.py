"""Freeze an audit-ready geographic yes/no sample for the pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from georeset_text_label_benchmark.output_schema import LABELLED_SCHEMA
from georeset_text_label_benchmark.pilot.geo_sampling import (
    H3_RESOLUTION,
    h3_cell_of,
    h3_centre_of,
    sample_id_of,
    select_geographic_sample,
)
from georeset_text_label_benchmark.pilot.protocol import (
    CANDIDATE_LABELS_SHA256,
    SAMPLE_SEED,
    SAMPLE_SIZE,
)
from georeset_text_label_benchmark.pilot.sampling import build_candidate_labels

LABELLED_NAME = "labelled-eunis.parquet"
PIPELINE_MANIFEST_NAME = "pipeline-manifest.json"
SOURCE_COLUMNS = tuple(LABELLED_SCHEMA.names)
SELECTION_METHOD = (
    "H3 resolution 3 cells with maximin spacing and a seeded tie-break; one sentence per "
    "cell; 50 yes and 50 no on disjoint cells; English only; one occurrence per exact "
    "sentence hash and per polygon; occurrences ordered by SHA256(seed:occurrence)."
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
EXPECTED_CANDIDATES = 158


@dataclass(frozen=True)
class _FreezeInputs:
    manifest: Mapping[str, Any]
    labelled_hash: str
    pipeline_hash: str
    candidate_hash: str


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
    labelled_parquet: Path,
    candidate_csv: Path,
    pipeline_manifest: Path,
    expected_candidate_sha256: str | None,
) -> tuple[str, str, Mapping[str, Any]]:
    labelled_hash = sha256_file(labelled_parquet)
    manifest = _read_pipeline_manifest(pipeline_manifest)
    if manifest["artifact_sha256"].get(LABELLED_NAME) != labelled_hash:
        raise ValueError("labelled-eunis.parquet SHA-256 does not match the pipeline manifest")
    candidate_hash = sha256_file(candidate_csv)
    if expected_candidate_sha256 is not None and candidate_hash != expected_candidate_sha256:
        raise ValueError("candidate CSV SHA-256 does not match the pinned input")
    return labelled_hash, candidate_hash, manifest


def _read_pipeline_manifest(path: Path) -> Mapping[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("pipeline manifest.json is unreadable") from error
    if not isinstance(manifest, Mapping) or not isinstance(
        manifest.get("artifact_sha256"), Mapping
    ):
        raise ValueError("pipeline manifest.json must record artifact checksums")
    return manifest


def _validate_candidate_count(candidates: Sequence[Mapping[str, Any]]) -> None:
    if len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError(f"expected {EXPECTED_CANDIDATES} candidate classes")


def freeze_sample(
    labelled_parquet: Path,
    candidate_csv: Path,
    pipeline_manifest: Path,
    output_dir: Path,
    *,
    expected_candidate_sha256: str | None = CANDIDATE_LABELS_SHA256,
    per_group: int = SAMPLE_SIZE // 2,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """Write the geographic yes/no sample and its candidate table before inference."""
    if output_dir.exists():
        raise FileExistsError(f"pilot directory already exists: {output_dir}")
    labelled_hash, candidate_hash, manifest = _validate_freeze_inputs(
        labelled_parquet, candidate_csv, pipeline_manifest, expected_candidate_sha256
    )
    source_rows, _ = _read_source_rows(labelled_parquet)
    candidates = build_candidate_labels(source_rows, _read_taxonomy(candidate_csv))
    _validate_candidate_count(candidates)
    selected = [
        {**row, "sample_id": sample_id_of(row)}
        for row in select_geographic_sample(
            source_rows,
            cell_of=h3_cell_of,
            centre_of=h3_centre_of,
            per_group=per_group,
            seed=seed,
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(candidate_csv, output_dir / "candidate_labels.csv")
    sample = _frozen_sample(
        _FreezeInputs(
            manifest=manifest,
            labelled_hash=labelled_hash,
            pipeline_hash=sha256_file(pipeline_manifest),
            candidate_hash=candidate_hash,
        ),
        source_rows,
        candidates,
        selected,
        per_group,
        seed,
    )
    _write_json_exclusive(output_dir / "frozen_sample.json", sample)
    _write_json_exclusive(
        output_dir / "manifest.json",
        _freeze_manifest(
            output_dir, sample, labelled_hash, sample["source"]["pipeline_manifest_sha256"]
        ),
    )
    return _freeze_summary(output_dir, per_group * 2, len(candidates), sample)


def _selection_metadata(
    selected: Sequence[Mapping[str, Any]], per_group: int, seed: int
) -> dict[str, Any]:
    return {
        "method": SELECTION_METHOD,
        "seed": seed,
        "per_group": per_group,
        "sample_size": 2 * per_group,
        "decision_counts": dict(sorted(Counter(row["decision"] for row in selected).items())),
        "h3_resolution": H3_RESOLUTION,
        "cell_count": len({row["h3_cell"] for row in selected}),
        "sample_ids_sha256": _sha256_json([row["sample_id"] for row in selected]),
        **_unique_counts(selected),
    }


def _unique_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "unique_sentence_hash_count": len({row["text_sha256"] for row in rows}),
        "unique_polygon_count": len(
            {(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in rows}
        ),
    }


def _frozen_source(
    manifest: Mapping[str, Any], labelled_hash: str, pipeline_hash: str
) -> dict[str, Any]:
    return {
        "file": LABELLED_NAME,
        "sha256": labelled_hash,
        "pipeline_manifest": PIPELINE_MANIFEST_NAME,
        "pipeline_manifest_sha256": pipeline_hash,
        "computation_commit": manifest.get("computation_commit"),
        "validation_commit": manifest.get("validation_commit"),
        "input_snapshots": manifest.get("input_snapshots"),
        "license": "OpenStreetMap ODbL-1.0; European Environment Agency CC-BY-4.0",
    }


def _frozen_sample(
    inputs: _FreezeInputs,
    source_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    per_group: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "source": _frozen_source(inputs.manifest, inputs.labelled_hash, inputs.pipeline_hash),
        "source_coverage": _sample_metadata(source_rows),
        "selection": _selection_metadata(selected, per_group, seed),
        "candidate_labels": {
            "file": "candidate_labels.csv",
            "sha256": inputs.candidate_hash,
            "count": len(candidates),
            "codes": [row["eunis_code"] for row in candidates],
            "text_method": "Exact pinned EUNIS English name, newline, exact EEA description.",
        },
        "selected_rows": list(selected),
    }


def _freeze_manifest(
    output_dir: Path, sample: Mapping[str, Any], labelled_hash: str, pipeline_hash: str
) -> dict[str, Any]:
    selection = sample["selection"]
    return {
        "format_version": 2,
        "method": selection["method"],
        "seed": selection["seed"],
        "per_group": selection["per_group"],
        "sample_size": selection["sample_size"],
        "sample_ids_sha256": selection["sample_ids_sha256"],
        "source_sha256": {LABELLED_NAME: labelled_hash, PIPELINE_MANIFEST_NAME: pipeline_hash},
        "outputs_sha256": {
            name: sha256_file(output_dir / name)
            for name in ("frozen_sample.json", "candidate_labels.csv")
        },
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
    if source.get("file") != LABELLED_NAME:
        raise ValueError("frozen sample is not from the labelled EUNIS pool")
    checksums = (source.get("sha256"), source.get("pipeline_manifest_sha256"))
    if not all(_is_sha256(value) for value in checksums):
        raise ValueError("frozen sample source checksums are missing")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


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
    _require_balanced_decisions(rows, selection["per_group"])
    _require_cell_centres(rows)


def _require_balanced_decisions(rows: Sequence[Mapping[str, Any]], per_group: int) -> None:
    counts = Counter(row["decision"] for row in rows)
    if counts.get("yes") != per_group or counts.get("no") != per_group:
        raise ValueError(f"frozen sample must have {per_group} yes and {per_group} no")


def _require_cell_centres(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if h3_centre_of(row["h3_cell"]) != (row["cell_centre_lat"], row["cell_centre_lon"]):
            raise ValueError("frozen sample H3 cell does not match its centre")


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
