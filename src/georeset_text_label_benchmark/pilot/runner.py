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
from typing import Any, TypeGuard

import pyarrow.parquet as pq

from georeset_text_label_benchmark.output_schema import LABELLED_SCHEMA
from georeset_text_label_benchmark.pilot.geo_sampling import (
    GROUPS,
    H3_RESOLUTION,
    extend_region_balanced_sample,
    h3_cell_of,
    h3_centre_of,
    sample_id_of,
    select_geographic_sample,
)
from georeset_text_label_benchmark.pilot.m49 import (
    M49_MINIMUM_ONLY_STRATA,
    M49_QUOTA_TIE_BREAK_ORDER,
    M49_UNASSIGNED,
    M49Assignment,
    m49_assignment_for_row,
    m49_boundary_provenance,
)
from georeset_text_label_benchmark.pilot.protocol import (
    CANDIDATE_LABELS_SHA256,
    EXPECTED_BASE_FROZEN_SAMPLE_SHA256,
    EXPECTED_BASE_MANIFEST_SHA256,
    EXPECTED_BASE_PIPELINE_MANIFEST_SHA256,
    EXPECTED_BASE_SELECTED_ROWS_SHA256,
    EXPECTED_LABELLED_POOL_SHA256,
    EXPECTED_SAMPLE_IDS_SHA256,
    SAMPLE_SEED,
    SAMPLE_SIZE,
)
from georeset_text_label_benchmark.pilot.sampling import build_candidate_labels

LABELLED_NAME = "labelled-eunis.parquet"
PIPELINE_MANIFEST_NAME = "pipeline-manifest.json"
SOURCE_COLUMNS = tuple(LABELLED_SCHEMA.names)
EXTENDED_SELECTION_METHOD = (
    "UN M49 subregion strata from polygon-bbox midpoints assigned against Natural Earth "
    "1:10m Admin-0 geometry; unassigned, unmapped, or ambiguous points remain explicit; "
    "each candidate H3 cell may connect to multiple regions; a bipartite max-flow check "
    "constrains the common yes/no quota vector; UNASSIGNED stays at the common prefix "
    "minimum while assigned regions balance as capacities allow with M49-code tie-breaks; "
    "maximin spacing selects cells within each stratum from stored base-cell centres; one sentence per "
    "cell within each decision group; yes and no may share a cell; immutable 50 yes and 50 "
    "no published pilot prefix; English only; one occurrence per exact sentence hash and "
    "per polygon; occurrences ordered by SHA256(seed:occurrence)."
)
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
    pipeline_manifest_provenance: Mapping[str, Any] | None = None


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
    extend_from: Path | None = None,
    allow_reconstructed_pipeline_manifest: bool = False,
) -> dict[str, Any]:
    """Write the geographic yes/no sample and its candidate table before inference."""
    _validate_freeze_request(output_dir, extend_from, allow_reconstructed_pipeline_manifest)
    labelled_hash, candidate_hash, manifest = _validate_freeze_inputs(
        labelled_parquet, candidate_csv, pipeline_manifest, expected_candidate_sha256
    )
    pipeline_hash = sha256_file(pipeline_manifest)
    all_rows, _ = _read_source_rows(labelled_parquet)
    taxonomy = _read_taxonomy(candidate_csv)
    source_rows = [row for row in all_rows if row["eunis_code"] in taxonomy]
    candidates = build_candidate_labels(source_rows, taxonomy)
    _validate_candidate_count(candidates)
    selected, method, geographic_stratification = _select_frozen_rows(
        source_rows,
        per_group,
        seed,
        extend_from,
        labelled_hash,
        pipeline_hash,
        allow_reconstructed_pipeline_manifest=allow_reconstructed_pipeline_manifest,
    )
    sample = _frozen_sample(
        _FreezeInputs(
            manifest=manifest,
            labelled_hash=labelled_hash,
            pipeline_hash=pipeline_hash,
            candidate_hash=candidate_hash,
            pipeline_manifest_provenance=_extension_manifest_provenance(
                extend_from,
                allow_reconstructed_pipeline_manifest,
                pipeline_hash,
                labelled_hash,
            ),
        ),
        source_rows,
        candidates,
        selected,
        per_group,
        seed,
        method,
        geographic_stratification,
    )
    sample["source_coverage"]["rows_outside_candidate_vocabulary"] = len(all_rows) - len(
        source_rows
    )
    _write_frozen_artifacts(output_dir, candidate_csv, sample, labelled_hash)
    return _freeze_summary(output_dir, per_group * 2, len(candidates), sample)


def _validate_freeze_request(
    output_dir: Path,
    extend_from: Path | None,
    allow_reconstructed_pipeline_manifest: bool,
) -> None:
    if allow_reconstructed_pipeline_manifest and extend_from is None:
        raise ValueError("reconstructed pipeline manifest mode requires --extend-from")
    if output_dir.exists():
        raise FileExistsError(f"pilot directory already exists: {output_dir}")


def _extension_manifest_provenance(
    extend_from: Path | None,
    allow_reconstructed_pipeline_manifest: bool,
    pipeline_hash: str,
    labelled_hash: str,
) -> Mapping[str, Any] | None:
    if (
        extend_from is not None
        and allow_reconstructed_pipeline_manifest
        and pipeline_hash != EXPECTED_BASE_PIPELINE_MANIFEST_SHA256
    ):
        return _reconstructed_manifest_provenance(pipeline_hash, labelled_hash)
    return None


def _select_frozen_rows(
    source_rows: Sequence[Mapping[str, Any]],
    per_group: int,
    seed: int,
    extend_from: Path | None,
    labelled_hash: str,
    pipeline_hash: str,
    *,
    allow_reconstructed_pipeline_manifest: bool,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    if extend_from is None:
        selected = select_geographic_sample(
            source_rows,
            cell_of=h3_cell_of,
            centre_of=h3_centre_of,
            per_group=per_group,
            seed=seed,
        )
        return (
            [{**row, "sample_id": sample_id_of(row)} for row in selected],
            SELECTION_METHOD,
            None,
        )
    base = _read_base_sample(
        extend_from,
        per_group,
        seed,
        expected_labelled_sha256=labelled_hash,
        expected_pipeline_manifest_sha256=pipeline_hash,
        allow_reconstructed_pipeline_manifest=allow_reconstructed_pipeline_manifest,
    )
    added, stratum_metadata = extend_region_balanced_sample(
        source_rows,
        cell_of=h3_cell_of,
        centre_of=h3_centre_of,
        stratum_of=lambda row: m49_assignment_for_row(dict(row)).stratum,
        existing=base,
        per_group=per_group,
        seed=seed,
        minimum_only_strata=M49_MINIMUM_ONLY_STRATA,
        quota_tie_break_order=M49_QUOTA_TIE_BREAK_ORDER,
    )
    additions = [{**row, "sample_id": sample_id_of(row)} for row in added]
    selected = [*base, *additions]
    return (
        selected,
        EXTENDED_SELECTION_METHOD,
        _extension_geographic_metadata(selected, stratum_metadata),
    )


def _extension_geographic_metadata(
    selected: Sequence[Mapping[str, Any]], stratum_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    assignments, region_names, status_counts, selected_counts = _collect_m49_selection_metadata(
        selected
    )
    if selected_counts != stratum_metadata["selected_counts_by_decision"]:
        raise ValueError("M49 selected region counts changed during manifest construction")
    quotas = stratum_metadata["target_counts_per_group"]
    return {
        **m49_boundary_provenance(),
        **stratum_metadata,
        "quota_method": (
            "M49-code-ordered common-quota balancing with bipartite max-flow feasibility "
            "for shared cell-to-region capacities; seeded SHA-256 resolves candidate and "
            "cell ties"
        ),
        "region_names_by_code": dict(sorted(region_names.items())),
        "eligible_subregion_codes": sorted(code for code in quotas if code != M49_UNASSIGNED),
        "assignments_by_sample_id": assignments,
        "assignment_status_counts_by_decision": _format_counter_groups(status_counts),
    }


def _collect_m49_selection_metadata(
    selected: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, str]], dict[str, str], dict[str, Counter[str]], dict[str, dict[str, int]]
]:
    assignments = []
    region_names = {M49_UNASSIGNED: "UNASSIGNED"}
    status_counts = {group: Counter() for group in GROUPS}
    selected_counts = {group: Counter() for group in GROUPS}
    for row in selected:
        assignment = m49_assignment_for_row(dict(row))
        assignments.append(_m49_assignment_record(row, assignment))
        _record_m49_assignment(
            row["decision"], assignment, region_names, status_counts, selected_counts
        )
    return (
        assignments,
        dict(sorted(region_names.items())),
        status_counts,
        _format_counter_groups(selected_counts),
    )


def _m49_assignment_record(row: Mapping[str, Any], assignment: M49Assignment) -> dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "stratum": assignment.stratum,
        "status": assignment.status,
    }


def _record_m49_assignment(
    group: str,
    assignment: M49Assignment,
    region_names: dict[str, str],
    status_counts: dict[str, Counter[str]],
    selected_counts: dict[str, Counter[str]],
) -> None:
    status_counts[group][assignment.status] += 1
    selected_counts[group][assignment.stratum] += 1
    if assignment.subregion_name is not None:
        region_names[assignment.stratum] = assignment.subregion_name


def _format_counter_groups(counts: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {group: dict(sorted(counts[group].items())) for group in GROUPS}


def _write_frozen_artifacts(
    output_dir: Path,
    candidate_csv: Path,
    sample: Mapping[str, Any],
    labelled_hash: str,
) -> None:
    output_dir.mkdir(parents=True)
    shutil.copyfile(candidate_csv, output_dir / "candidate_labels.csv")
    _write_json_exclusive(output_dir / "frozen_sample.json", sample)
    _write_json_exclusive(
        output_dir / "manifest.json",
        _freeze_manifest(
            output_dir, sample, labelled_hash, sample["source"]["pipeline_manifest_sha256"]
        ),
    )


def _read_base_sample(
    path: Path,
    per_group: int,
    seed: int,
    *,
    expected_labelled_sha256: str,
    expected_pipeline_manifest_sha256: str,
    allow_reconstructed_pipeline_manifest: bool = False,
) -> list[dict[str, Any]]:
    """Load the published pilot after checking its identity and source provenance."""
    sample = _read_base_sample_json(path)
    selection = sample["selection"]
    _validate_base_selection(selection, per_group, seed)
    _validate_base_source(
        sample,
        expected_labelled_sha256,
        expected_pipeline_manifest_sha256,
        allow_reconstructed_pipeline_manifest=allow_reconstructed_pipeline_manifest,
    )
    rows = sample.get("selected_rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("base frozen sample must contain selected row objects")
    _validate_base_manifest(path, sample)
    _validate_frozen_rows(rows, selection, pinned_base=True)
    return rows


def _read_base_sample_json(path: Path) -> Mapping[str, Any]:
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("base frozen sample is unreadable") from error
    if not isinstance(sample, Mapping) or not isinstance(sample.get("selection"), Mapping):
        raise ValueError("base frozen sample must contain a selection object")
    return sample


def _validate_base_selection(selection: Mapping[str, Any], per_group: int, seed: int) -> None:
    _validate_base_identity(selection, seed)
    _validate_base_group_size(selection)
    if selection["per_group"] > per_group:
        raise ValueError("requested per_group is smaller than the base sample")


def _validate_base_identity(selection: Mapping[str, Any], seed: int) -> None:
    expected = {
        "method": SELECTION_METHOD,
        "seed": seed,
        "sample_size": SAMPLE_SIZE,
        "sample_ids_sha256": EXPECTED_SAMPLE_IDS_SHA256,
    }
    if (
        not _is_integer_metadata(selection.get("seed"))
        or not _is_integer_metadata(selection.get("sample_size"))
        or any(selection.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("base sample must be the published pilot sample with the same seed")


def _validate_base_group_size(selection: Mapping[str, Any]) -> None:
    per_group = selection.get("per_group")
    if (
        isinstance(per_group, bool)
        or not isinstance(per_group, int)
        or 2 * per_group != SAMPLE_SIZE
    ):
        raise ValueError("base sample must contain the published 50/50 sample")


def _validate_base_source(
    sample: Mapping[str, Any],
    expected_labelled_sha256: str,
    expected_pipeline_manifest_sha256: str,
    *,
    allow_reconstructed_pipeline_manifest: bool = False,
) -> None:
    source = sample.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("base frozen sample must contain source provenance")
    _validate_frozen_source(source)
    if source["sha256"] != expected_labelled_sha256:
        raise ValueError("base sample and labelled pool must have the same SHA-256")
    _validate_base_pipeline_hash(
        source,
        expected_labelled_sha256,
        expected_pipeline_manifest_sha256,
        allow_reconstructed_pipeline_manifest,
    )


def _validate_base_pipeline_hash(
    source: Mapping[str, Any],
    expected_labelled_sha256: str,
    expected_pipeline_manifest_sha256: str,
    allow_reconstructed_pipeline_manifest: bool,
) -> None:
    if source["pipeline_manifest_sha256"] != expected_pipeline_manifest_sha256:
        if not allow_reconstructed_pipeline_manifest:
            raise ValueError("base sample and pipeline manifest must have the same SHA-256")
        _validate_reconstructed_base_identity(source, expected_labelled_sha256)


def _validate_reconstructed_base_identity(
    source: Mapping[str, Any], expected_labelled_sha256: str
) -> None:
    if source["pipeline_manifest_sha256"] != EXPECTED_BASE_PIPELINE_MANIFEST_SHA256:
        raise ValueError(
            "base sample does not record the pinned historical pipeline manifest SHA-256"
        )
    if (
        expected_labelled_sha256 != EXPECTED_LABELLED_POOL_SHA256
        or source["sha256"] != EXPECTED_LABELLED_POOL_SHA256
    ):
        raise ValueError(
            "reconstructed pipeline manifest extension requires the pinned labelled pool SHA-256"
        )


def _validate_base_manifest(path: Path, sample: Mapping[str, Any]) -> None:
    sample_hash = sha256_file(path)
    if sample_hash != EXPECTED_BASE_FROZEN_SAMPLE_SHA256:
        raise ValueError("base frozen sample checksum does not match the published pilot")
    manifest_path = path.with_name("manifest.json")
    manifest = _read_base_manifest(manifest_path)
    _validate_base_manifest_contents(manifest, sample, sample_hash)


def _read_base_manifest(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError("base frozen manifest.json is required to verify the published sample")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("base frozen manifest.json is unreadable") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("base frozen manifest.json must contain a JSON object")
    if sha256_file(path) != EXPECTED_BASE_MANIFEST_SHA256:
        raise ValueError("base frozen manifest checksum does not match the published pilot")
    return manifest


def _validate_base_manifest_contents(
    manifest: Mapping[str, Any], sample: Mapping[str, Any], sample_hash: str
) -> None:
    _validate_base_manifest_outputs(manifest.get("outputs_sha256"), sample_hash)
    _validate_base_manifest_sources(manifest.get("source_sha256"), sample["source"])
    _validate_base_manifest_selection(manifest, sample["selection"])


def _validate_base_manifest_outputs(outputs: Any, sample_hash: str) -> None:
    if not isinstance(outputs, Mapping):
        raise ValueError("base frozen manifest is missing input or output checksums")
    if outputs.get("frozen_sample.json") != sample_hash:
        raise ValueError("base frozen manifest checksums do not match the published sample")
    if outputs.get("candidate_labels.csv") != CANDIDATE_LABELS_SHA256:
        raise ValueError("base frozen manifest checksums do not match the published sample")


def _validate_base_manifest_sources(source_hashes: Any, source: Mapping[str, Any]) -> None:
    if not isinstance(source_hashes, Mapping):
        raise ValueError("base frozen manifest is missing input or output checksums")
    if source_hashes.get(LABELLED_NAME) != source["sha256"]:
        raise ValueError("base frozen manifest checksums do not match the published sample")
    if source_hashes.get(PIPELINE_MANIFEST_NAME) != source["pipeline_manifest_sha256"]:
        raise ValueError("base frozen manifest checksums do not match the published sample")


def _validate_base_manifest_selection(
    manifest: Mapping[str, Any], selection: Mapping[str, Any]
) -> None:
    if any(
        manifest.get(key) != selection.get(key)
        for key in ("method", "seed", "per_group", "sample_size", "sample_ids_sha256")
    ):
        raise ValueError("base frozen manifest selection does not match the published sample")


def _selection_metadata(
    selected: Sequence[Mapping[str, Any]],
    per_group: int,
    seed: int,
    method: str,
    geographic_stratification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "method": method,
        "seed": seed,
        "per_group": per_group,
        "sample_size": 2 * per_group,
        "decision_counts": dict(sorted(Counter(row["decision"] for row in selected).items())),
        "h3_resolution": H3_RESOLUTION,
        "cell_count": len({row["h3_cell"] for row in selected}),
        "sample_ids_sha256": _sha256_json([row["sample_id"] for row in selected]),
        **_unique_counts(selected),
    }
    if geographic_stratification is not None:
        metadata["geographic_stratification"] = dict(geographic_stratification)
    return metadata


def _unique_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "unique_sentence_hash_count": len({row["text_sha256"] for row in rows}),
        "unique_polygon_count": len(
            {(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in rows}
        ),
    }


def _frozen_source(
    manifest: Mapping[str, Any],
    labelled_hash: str,
    pipeline_hash: str,
    pipeline_manifest_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = {
        "file": LABELLED_NAME,
        "sha256": labelled_hash,
        "pipeline_manifest": PIPELINE_MANIFEST_NAME,
        "pipeline_manifest_sha256": pipeline_hash,
        "computation_commit": manifest.get("computation_commit"),
        "validation_commit": manifest.get("validation_commit"),
        "input_snapshots": manifest.get("input_snapshots"),
        "license": "OpenStreetMap ODbL-1.0; European Environment Agency CC-BY-4.0",
    }
    if pipeline_manifest_provenance is not None:
        source["pipeline_manifest_provenance"] = dict(pipeline_manifest_provenance)
    return source


def _reconstructed_manifest_provenance(
    extension_manifest_sha256: str, labelled_pool_sha256: str
) -> dict[str, str]:
    if labelled_pool_sha256 != EXPECTED_LABELLED_POOL_SHA256:
        raise ValueError(
            "reconstructed pipeline manifest extension requires the pinned labelled pool SHA-256"
        )
    if extension_manifest_sha256 == EXPECTED_BASE_PIPELINE_MANIFEST_SHA256:
        raise ValueError("reconstructed pipeline manifest must differ from the historical manifest")
    return {
        "mode": "reconstructed_manifest_same_labelled_pool",
        "historical_pipeline_manifest_sha256": EXPECTED_BASE_PIPELINE_MANIFEST_SHA256,
        "extension_pipeline_manifest_sha256": extension_manifest_sha256,
        "labelled_pool_sha256": labelled_pool_sha256,
        "base_frozen_sample_sha256": EXPECTED_BASE_FROZEN_SAMPLE_SHA256,
        "base_frozen_manifest_sha256": EXPECTED_BASE_MANIFEST_SHA256,
        "base_selected_rows_sha256": EXPECTED_BASE_SELECTED_ROWS_SHA256,
    }


def _frozen_sample(
    inputs: _FreezeInputs,
    source_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    per_group: int,
    seed: int,
    method: str,
    geographic_stratification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "source": _frozen_source(
            inputs.manifest,
            inputs.labelled_hash,
            inputs.pipeline_hash,
            inputs.pipeline_manifest_provenance,
        ),
        "source_coverage": _sample_metadata(source_rows),
        "selection": _selection_metadata(
            selected, per_group, seed, method, geographic_stratification
        ),
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
    manifest = {
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
    if "geographic_stratification" in selection:
        manifest["geographic_stratification"] = selection["geographic_stratification"]
    return manifest


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
    if "pipeline_manifest_provenance" in source:
        _validate_reconstructed_manifest_provenance(source)


def _validate_reconstructed_manifest_provenance(source: Mapping[str, Any]) -> None:
    provenance = source.get("pipeline_manifest_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("frozen sample reconstructed-manifest provenance is inconsistent")
    expected = {
        "mode": "reconstructed_manifest_same_labelled_pool",
        "historical_pipeline_manifest_sha256": EXPECTED_BASE_PIPELINE_MANIFEST_SHA256,
        "extension_pipeline_manifest_sha256": source["pipeline_manifest_sha256"],
        "labelled_pool_sha256": source["sha256"],
        "base_frozen_sample_sha256": EXPECTED_BASE_FROZEN_SAMPLE_SHA256,
        "base_frozen_manifest_sha256": EXPECTED_BASE_MANIFEST_SHA256,
        "base_selected_rows_sha256": EXPECTED_BASE_SELECTED_ROWS_SHA256,
    }
    if (
        dict(provenance) != expected
        or source["sha256"] != EXPECTED_LABELLED_POOL_SHA256
        or source["pipeline_manifest_sha256"] == EXPECTED_BASE_PIPELINE_MANIFEST_SHA256
    ):
        raise ValueError("frozen sample reconstructed-manifest provenance is inconsistent")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _validate_frozen_rows(
    rows: list[dict[str, Any]],
    selection: Mapping[str, Any],
    *,
    pinned_base: bool = False,
) -> None:
    _validate_frozen_row_count(rows, selection)
    _validate_frozen_row_contents(rows)
    ids = [row["sample_id"] for row in rows]
    _require_unique_frozen_values(ids)
    _require_unique_frozen_values([row["text_sha256"] for row in rows])
    polygons = [(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in rows]
    _require_unique_frozen_values(polygons)
    _validate_frozen_ids(ids, selection)
    _validate_frozen_selection(rows, selection, pinned_base=pinned_base)


def _validate_frozen_row_count(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> None:
    if len(rows) != selection["sample_size"]:
        raise ValueError("frozen sample row count does not match its declaration")


def _validate_frozen_row_contents(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        _validate_frozen_row_content(row)


def _validate_frozen_row_content(row: Mapping[str, Any]) -> None:
    sentence = row.get("sentence")
    if not isinstance(sentence, str):
        raise ValueError("frozen sample sentence must be text")
    text_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
    if row.get("text_sha256") != text_hash:
        raise ValueError("frozen sample sentence hash mismatch")
    try:
        expected_id = sample_id_of(row)
    except (KeyError, TypeError) as error:
        raise ValueError("frozen sample row identity is incomplete") from error
    if row.get("sample_id") != expected_id:
        raise ValueError("frozen sample ID does not match its row identity")


def _require_unique_frozen_values(values: Sequence[Any]) -> None:
    if len(set(values)) != len(values):
        raise ValueError("frozen sample must have unique IDs, sentence hashes, and polygons")


def _validate_frozen_ids(ids: Sequence[str], selection: Mapping[str, Any]) -> None:
    if _sha256_json(ids) != selection["sample_ids_sha256"]:
        raise ValueError("frozen sample ID checksum mismatch")


def _validate_frozen_selection(
    rows: list[dict[str, Any]],
    selection: Mapping[str, Any],
    *,
    pinned_base: bool = False,
) -> None:
    _require_balanced_decisions(rows, selection["per_group"])
    _validate_selection_metadata(rows, selection)
    _validate_m49_stratification(rows, selection)
    _validate_frozen_cell_centres(rows, selection, pinned_base)
    _require_unique_group_cells(rows, selection["method"])


def _validate_m49_stratification(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> None:
    geography = selection.get("geographic_stratification")
    if selection.get("method") != EXTENDED_SELECTION_METHOD:
        if geography is not None:
            raise ValueError("non-extended sample cannot declare M49 extension quotas")
        return
    if not isinstance(geography, Mapping):
        raise ValueError("extended frozen sample must declare M49 geographic stratification")
    _validate_m49_provenance(geography)
    _validate_m49_quota_protocol(geography)
    assignments = _validated_m49_assignments(rows, geography)
    region_counts, status_counts, region_names = _m49_assignment_counts(rows, assignments)
    _validate_m49_recorded_counts(geography, region_counts, status_counts, region_names)
    _validate_m49_quotas(selection, geography, rows, assignments, region_counts)


def _validate_m49_provenance(geography: Mapping[str, Any]) -> None:
    expected = m49_boundary_provenance()
    if any(geography.get(key) != value for key, value in expected.items()):
        raise ValueError("extended frozen sample M49 boundary provenance is inconsistent")
    if geography.get("quota_method") != (
        "M49-code-ordered common-quota balancing with bipartite max-flow feasibility for "
        "shared cell-to-region capacities; seeded SHA-256 resolves candidate and cell ties"
    ):
        raise ValueError("extended frozen sample M49 quota method is unsupported")


def _validate_m49_quota_protocol(geography: Mapping[str, Any]) -> None:
    if geography.get("minimum_only_strata") != sorted(M49_MINIMUM_ONLY_STRATA):
        raise ValueError("extended frozen sample M49 quota policy is unsupported")
    if geography.get("quota_tie_break_order") != list(M49_QUOTA_TIE_BREAK_ORDER):
        raise ValueError("extended frozen sample M49 quota policy is unsupported")


def _validated_m49_assignments(
    rows: Sequence[Mapping[str, Any]], geography: Mapping[str, Any]
) -> list[M49Assignment]:
    assignments = geography.get("assignments_by_sample_id")
    if not isinstance(assignments, list) or len(assignments) != len(rows):
        raise ValueError("extended frozen sample M49 assignments do not cover every row")
    validated = []
    for row, assignment in zip(rows, assignments, strict=True):
        validated.append(_validate_one_m49_assignment(row, assignment))
    return validated


def _validate_one_m49_assignment(row: Mapping[str, Any], assignment: Any) -> M49Assignment:
    if not isinstance(assignment, Mapping) or assignment.get("sample_id") != row.get("sample_id"):
        raise ValueError("extended frozen sample M49 assignment order does not match its rows")
    expected = m49_assignment_for_row(dict(row))
    if assignment.get("stratum") != expected.stratum or assignment.get("status") != expected.status:
        raise ValueError("extended frozen sample M49 assignment does not match its coordinates")
    return expected


def _m49_assignment_counts(
    rows: Sequence[Mapping[str, Any]], assignments: Sequence[M49Assignment]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, str]]:
    region_counts = {group: Counter() for group in GROUPS}
    status_counts = {group: Counter() for group in GROUPS}
    region_names = {M49_UNASSIGNED: "UNASSIGNED"}
    for row, assignment in zip(rows, assignments, strict=True):
        _count_m49_assignment(
            row["decision"], assignment, region_counts, status_counts, region_names
        )
    return (
        _format_counter_groups(region_counts),
        _format_counter_groups(status_counts),
        dict(sorted(region_names.items())),
    )


def _validate_m49_recorded_counts(
    geography: Mapping[str, Any],
    region_counts: Mapping[str, Mapping[str, int]],
    status_counts: Mapping[str, Mapping[str, int]],
    region_names: Mapping[str, str],
) -> None:
    _require_m49_record(geography, "selected_counts_by_decision", region_counts, "selected counts")
    _require_m49_record(
        geography, "assignment_status_counts_by_decision", status_counts, "status counts"
    )
    _require_m49_record(geography, "region_names_by_code", region_names, "region names")


def _count_m49_assignment(
    group: str,
    assignment: M49Assignment,
    region_counts: dict[str, Counter[str]],
    status_counts: dict[str, Counter[str]],
    region_names: dict[str, str],
) -> None:
    if not assignment.stratum or not assignment.status:
        raise ValueError("extended frozen sample M49 assignment fields are invalid")
    region_counts[group][assignment.stratum] += 1
    status_counts[group][assignment.status] += 1
    if assignment.subregion_name is not None:
        region_names[assignment.stratum] = assignment.subregion_name


def _require_m49_record(
    geography: Mapping[str, Any],
    field: str,
    expected: Any,
    description: str,
    *,
    expected_scope: str = "rows",
) -> None:
    if geography.get(field) != expected:
        raise ValueError(
            f"extended frozen sample M49 {description} do not match its {expected_scope}"
        )


def _validate_m49_quotas(
    selection: Mapping[str, Any],
    geography: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[M49Assignment],
    region_counts: Mapping[str, Mapping[str, int]],
) -> None:
    quotas, prefix, capacities = _m49_quota_mappings(geography)
    _validate_common_m49_quota(selection, quotas, region_counts)
    _validate_m49_capacity_records(geography, quotas, prefix, capacities)
    _validate_m49_unassigned_quota_minimum(quotas, prefix)
    expected_prefix = _format_counter_groups(
        _m49_counts_from_pairs(rows[:SAMPLE_SIZE], assignments[:SAMPLE_SIZE])
    )
    _require_m49_record(
        geography,
        "prefix_counts_by_decision",
        expected_prefix,
        "prefix counts",
        expected_scope="base rows",
    )
    _validate_m49_eligible_regions(geography, quotas)


def _m49_quota_mappings(
    geography: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    quotas = geography.get("target_counts_per_group")
    prefix = geography.get("prefix_counts_by_decision")
    capacities = geography.get("candidate_cell_capacity_by_decision")
    if (
        not isinstance(quotas, Mapping)
        or not isinstance(prefix, Mapping)
        or not isinstance(capacities, Mapping)
    ):
        raise ValueError("extended frozen sample M49 quota records are missing")
    return quotas, prefix, capacities


def _validate_m49_unassigned_quota_minimum(
    quotas: Mapping[str, Any], prefix: Mapping[str, Mapping[str, Any]]
) -> None:
    minimum = max(prefix[group].get(M49_UNASSIGNED, 0) for group in GROUPS)
    if quotas.get(M49_UNASSIGNED, 0) != minimum:
        raise ValueError(
            "extended frozen sample UNASSIGNED quota must equal the common prefix minimum"
        )


def _validate_common_m49_quota(
    selection: Mapping[str, Any],
    quotas: Mapping[str, Any],
    region_counts: Mapping[str, Mapping[str, int]],
) -> None:
    _require_integer_m49_quotas(quotas)
    _require_m49_quota_total(selection, quotas)
    _require_common_m49_group_counts(quotas, region_counts)


def _require_integer_m49_quotas(quotas: Mapping[str, Any]) -> None:
    if not all(_is_integer_metadata(value) for value in quotas.values()):
        raise ValueError("extended frozen sample does not meet its common M49 quota vector")


def _require_m49_quota_total(selection: Mapping[str, Any], quotas: Mapping[str, Any]) -> None:
    if sum(quotas.values()) != selection["per_group"]:
        raise ValueError("extended frozen sample does not meet its common M49 quota vector")


def _require_common_m49_group_counts(
    quotas: Mapping[str, Any], region_counts: Mapping[str, Mapping[str, int]]
) -> None:
    expected = _positive_m49_counts(quotas)
    for group in GROUPS:
        if _positive_m49_counts(region_counts[group]) != expected:
            raise ValueError("extended frozen sample does not meet its common M49 quota vector")


def _positive_m49_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {stratum: count for stratum, count in counts.items() if count > 0}


def _validate_m49_eligible_regions(geography: Mapping[str, Any], quotas: Mapping[str, Any]) -> None:
    expected = sorted(code for code in quotas if code != M49_UNASSIGNED)
    if geography.get("eligible_subregion_codes") != expected:
        raise ValueError("extended frozen sample M49 eligible region list is inconsistent")


def _m49_counts_from_pairs(
    rows: Sequence[Mapping[str, Any]], assignments: Sequence[M49Assignment]
) -> dict[str, Counter[str]]:
    counts = {group: Counter() for group in GROUPS}
    for row, assignment in zip(rows, assignments, strict=True):
        counts[row["decision"]][assignment.stratum] += 1
    return counts


def _validate_m49_capacity_records(
    geography: Mapping[str, Any],
    quotas: Mapping[str, Any],
    prefix: Mapping[str, Any],
    capacities: Mapping[str, Any],
) -> None:
    _validate_m49_capacity_shapes(prefix, capacities)
    _validate_m49_candidate_row_capacities(geography, capacities)
    _validate_m49_unique_cell_capacities(geography, capacities)
    _validate_m49_quota_bounds(quotas, prefix, capacities)


def _validate_m49_candidate_row_capacities(
    geography: Mapping[str, Any], cell_capacities: Mapping[str, Any]
) -> None:
    row_capacities = geography.get("candidate_row_capacity_by_decision")
    if not isinstance(row_capacities, Mapping):
        raise ValueError("extended frozen sample M49 candidate row capacities are invalid")
    for group in GROUPS:
        rows, cells = _m49_row_and_cell_capacities(group, row_capacities, cell_capacities)
        _validate_m49_group_row_capacity(rows, cells)


def _m49_row_and_cell_capacities(
    group: str, row_capacities: Mapping[str, Any], cell_capacities: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    rows = row_capacities.get(group)
    cells = cell_capacities.get(group)
    if not isinstance(rows, Mapping) or not isinstance(cells, Mapping):
        raise ValueError("extended frozen sample M49 candidate row capacities are invalid")
    return rows, cells


def _validate_m49_group_row_capacity(rows: Mapping[str, Any], cells: Mapping[str, Any]) -> None:
    _require_nonnegative_integer_m49_capacities(tuple(rows.values()))
    if any(rows.get(stratum, 0) < count for stratum, count in cells.items()):
        raise ValueError("extended frozen sample M49 row capacity is below its cell capacity")


def _validate_m49_unique_cell_capacities(
    geography: Mapping[str, Any], cell_capacities: Mapping[str, Any]
) -> None:
    unique = geography.get("candidate_unique_cell_count_by_decision")
    multi_region = geography.get("candidate_multi_region_cell_count_by_decision")
    if geography.get("cell_capacities_overlap_across_regions") is not True:
        raise ValueError("extended frozen sample M49 cell overlap provenance is missing")
    if not isinstance(unique, Mapping) or not isinstance(multi_region, Mapping):
        raise ValueError("extended frozen sample M49 unique cell capacities are invalid")
    for group in GROUPS:
        _validate_m49_group_cell_capacity(group, unique, multi_region, cell_capacities)


def _validate_m49_group_cell_capacity(
    group: str,
    unique: Mapping[str, Any],
    multi_region: Mapping[str, Any],
    cell_capacities: Mapping[str, Any],
) -> None:
    count = _validated_nonnegative_m49_count(unique.get(group))
    multi = _validated_nonnegative_m49_count(multi_region.get(group))
    regions = cell_capacities.get(group)
    if not isinstance(regions, Mapping):
        raise ValueError("extended frozen sample M49 unique cell capacities are invalid")
    _require_region_capacity_below_unique_cell_count(count, regions)
    _validate_m49_cell_membership_bounds(count, multi, regions)


def _validated_nonnegative_m49_count(value: Any) -> int:
    if not _is_integer_metadata(value) or value < 0:
        raise ValueError("extended frozen sample M49 unique cell capacities are invalid")
    return value


def _require_region_capacity_below_unique_cell_count(
    count: int, regions: Mapping[str, Any]
) -> None:
    if count < max(regions.values(), default=0):
        raise ValueError("extended frozen sample M49 unique cell capacities are invalid")


def _validate_m49_cell_membership_bounds(
    count: int, multi: int, regions: Mapping[str, Any]
) -> None:
    positive_strata = sum(value > 0 for value in regions.values())
    memberships = sum(regions.values())
    _require_multi_cell_count_within_total(multi, count)
    _require_region_membership_lower_bound(count, multi, memberships)
    _require_region_membership_upper_bound(count, multi, positive_strata, memberships)


def _require_multi_cell_count_within_total(multi: int, count: int) -> None:
    if multi > count:
        raise ValueError("extended frozen sample M49 unique cell capacities are invalid")


def _require_region_membership_lower_bound(count: int, multi: int, memberships: int) -> None:
    if memberships < count + multi:
        raise ValueError("extended frozen sample M49 unique cell capacities are invalid")


def _require_region_membership_upper_bound(
    count: int, multi: int, positive_strata: int, memberships: int
) -> None:
    maximum = count + multi * max(0, positive_strata - 1)
    if memberships > maximum:
        raise ValueError("extended frozen sample M49 unique cell capacities are invalid")


def _validate_m49_capacity_shapes(prefix: Mapping[str, Any], capacities: Mapping[str, Any]) -> None:
    for group in GROUPS:
        _validate_one_m49_capacity_shape(prefix, capacities, group)


def _validate_one_m49_capacity_shape(
    prefix: Mapping[str, Any], capacities: Mapping[str, Any], group: str
) -> None:
    prefix_counts, candidate_counts = _m49_capacity_group_mappings(prefix, capacities, group)
    values = (*prefix_counts.values(), *candidate_counts.values())
    _require_nonnegative_integer_m49_capacities(values)


def _m49_capacity_group_mappings(
    prefix: Mapping[str, Any], capacities: Mapping[str, Any], group: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    prefix_counts = prefix.get(group)
    candidate_counts = capacities.get(group)
    if not isinstance(prefix_counts, Mapping):
        raise ValueError("extended frozen sample M49 capacity records are invalid")
    if not isinstance(candidate_counts, Mapping):
        raise ValueError("extended frozen sample M49 capacity records are invalid")
    return prefix_counts, candidate_counts


def _require_nonnegative_integer_m49_capacities(values: Sequence[Any]) -> None:
    if any(not _is_integer_metadata(value) for value in values):
        raise ValueError("extended frozen sample M49 capacities must be non-negative integers")
    if any(value < 0 for value in values):
        raise ValueError("extended frozen sample M49 capacities must be non-negative integers")


def _validate_m49_quota_bounds(
    quotas: Mapping[str, Any], prefix: Mapping[str, Any], capacities: Mapping[str, Any]
) -> None:
    for stratum, quota in quotas.items():
        minimum = max(prefix[group].get(stratum, 0) for group in GROUPS)
        maximum = min(
            prefix[group].get(stratum, 0) + capacities[group].get(stratum, 0) for group in GROUPS
        )
        if not minimum <= quota <= maximum:
            raise ValueError("extended frozen sample M49 quota exceeds its recorded capacity")


def _validate_frozen_cell_centres(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], pinned_base: bool
) -> None:
    if selection.get("method") == EXTENDED_SELECTION_METHOD:
        _require_published_base_prefix(rows)
        _require_cell_centres(rows[SAMPLE_SIZE:])
    elif not pinned_base:
        _require_cell_centres(rows)


def _require_published_base_prefix(rows: Sequence[Mapping[str, Any]]) -> None:
    if (
        len(rows) < SAMPLE_SIZE
        or _sha256_json(rows[:SAMPLE_SIZE]) != EXPECTED_BASE_SELECTED_ROWS_SHA256
    ):
        raise ValueError("extended frozen sample base rows do not match the published pilot")


def _validate_selection_metadata(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> None:
    _require_supported_selection_method(selection)
    _require_declared_group_size(selection)
    _require_declared_seed(selection)
    _require_h3_resolution(selection)
    _require_declared_cell_count(rows, selection)
    _validate_selection_counts(rows, selection)


def _require_supported_selection_method(selection: Mapping[str, Any]) -> None:
    if selection.get("method") not in {SELECTION_METHOD, EXTENDED_SELECTION_METHOD}:
        raise ValueError("frozen sample selection method is not supported")


def _require_declared_group_size(selection: Mapping[str, Any]) -> None:
    per_group = selection.get("per_group")
    sample_size = selection.get("sample_size")
    if not _is_integer_metadata(per_group) or not _is_integer_metadata(sample_size):
        raise ValueError("frozen sample size does not match its per-group count")
    if sample_size != 2 * per_group:
        raise ValueError("frozen sample size does not match its per-group count")


def _require_declared_seed(selection: Mapping[str, Any]) -> None:
    if not _is_integer_metadata(selection.get("seed")):
        raise ValueError("frozen sample seed must be an integer")


def _is_integer_metadata(value: Any) -> TypeGuard[int]:
    return not isinstance(value, bool) and isinstance(value, int)


def _require_h3_resolution(selection: Mapping[str, Any]) -> None:
    if selection.get("h3_resolution") != H3_RESOLUTION:
        raise ValueError("frozen sample H3 resolution does not match the protocol")


def _require_declared_cell_count(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> None:
    if selection.get("cell_count") != _unique_cell_count(rows):
        raise ValueError("frozen sample cell count does not match its rows")


def _unique_cell_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({row["h3_cell"] for row in rows})


def _validate_selection_counts(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> None:
    expected = _selection_row_counts(rows)
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen sample counts do not match its rows")


def _selection_row_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "decision_counts": _decision_counts(rows),
        "unique_sentence_hash_count": _unique_sentence_hash_count(rows),
        "unique_polygon_count": _unique_polygon_count(rows),
    }


def _decision_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["decision"] for row in rows).items()))


def _unique_sentence_hash_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({row["text_sha256"] for row in rows})


def _unique_polygon_count(rows: Sequence[Mapping[str, Any]]) -> int:
    polygons = {(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in rows}
    return len(polygons)


def _require_unique_group_cells(rows: Sequence[Mapping[str, Any]], method: str) -> None:
    cells = _cells_by_decision(rows)
    _require_unique_cells(cells["yes"])
    _require_unique_cells(cells["no"])
    if method == SELECTION_METHOD:
        _require_disjoint_decision_cells(cells["yes"], cells["no"])


def _cells_by_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {group: [row["h3_cell"] for row in rows if row["decision"] == group] for group in GROUPS}


def _require_unique_cells(cells: Sequence[str]) -> None:
    if len(cells) != len(set(cells)):
        raise ValueError("frozen sample must have one sentence per cell within each group")


def _require_disjoint_decision_cells(yes_cells: Sequence[str], no_cells: Sequence[str]) -> None:
    if set(yes_cells).intersection(no_cells):
        raise ValueError("published pilot sample must use disjoint yes and no cells")


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
    pinned_base = sha256_file(frozen_path) == EXPECTED_BASE_FROZEN_SAMPLE_SHA256
    _validate_frozen_rows(rows, sample["selection"], pinned_base=pinned_base)
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
