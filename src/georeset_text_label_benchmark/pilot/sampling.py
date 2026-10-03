"""Deterministic selection and candidate construction for the pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

_ROW_FIELDS = (
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
_STRING_FIELDS = (
    "source_pbf",
    "osm_type",
    "description_identity",
    "tag_key",
    "sentence",
    "text_sha256",
    "eunis_code",
    "eunis_name",
)
_IDENTITY_FIELDS = (
    "source_pbf",
    "osm_type",
    "osm_id",
    "description_identity",
    "tag_key",
    "sentence_index",
    "text_sha256",
)


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_integer(row: Mapping[str, Any], field: str, minimum: int, error: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be a non-null integer")
    if value < minimum:
        raise ValueError(error)
    return value


def _validate_language(value: Any) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError("language_code must be a non-empty string or null")


def _validate_sentence_hash(sentence: str, text_hash: str) -> None:
    actual_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
    if text_hash != actual_hash:
        raise ValueError("sentence hash mismatch")


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {field: row.get(field) for field in _ROW_FIELDS}
    for field in _STRING_FIELDS:
        _required_string(clean, field)
    clean["osm_id"] = _required_integer(
        row, "osm_id", 1, "OSM IDs must be positive and sentence indices non-negative"
    )
    clean["sentence_index"] = _required_integer(
        row, "sentence_index", 0, "OSM IDs must be positive and sentence indices non-negative"
    )
    _validate_language(clean["language_code"])
    _validate_sentence_hash(
        _required_string(clean, "sentence"), _required_string(clean, "text_sha256")
    )
    return clean


def _sample_id(row: Mapping[str, Any]) -> str:
    identity = [row[field] for field in _IDENTITY_FIELDS]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deduplicate_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        clean = _validate_row(row)
        identifier = _sample_id(clean)
        existing = unique.get(identifier)
        if existing is not None and existing != clean:
            raise ValueError(f"conflicting duplicate occurrence: {identifier}")
        unique[identifier] = clean
    return unique


def _rank_rows(rows: Mapping[str, dict[str, Any]], seed: int) -> list[tuple[str, str]]:
    priorities = [
        (hashlib.sha256(f"{seed}:{identifier}".encode("ascii")).hexdigest(), identifier)
        for identifier in rows
    ]
    return sorted(priorities)


def _select_ranked(
    ranked_ids: Iterable[tuple[str, str]], rows: Mapping[str, dict[str, Any]], size: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    sentence_hashes: set[str] = set()
    polygon_keys: set[tuple[str, str, int]] = set()
    for _, identifier in ranked_ids:
        row = rows[identifier]
        polygon = (row["source_pbf"], row["osm_type"], row["osm_id"])
        if row["text_sha256"] in sentence_hashes or polygon in polygon_keys:
            continue
        sentence_hashes.add(row["text_sha256"])
        polygon_keys.add(polygon)
        selected.append({**row, "sample_id": identifier})
        if len(selected) == size:
            break
    return selected


def select_distinct_sample(
    rows: Iterable[Mapping[str, Any]], *, size: int, seed: int
) -> list[dict[str, Any]]:
    """Select deterministic rows with unique text hashes and polygons."""
    _validate_sample_parameters(size, seed)
    unique_rows = _deduplicate_rows(rows)
    chosen = _select_ranked(_rank_rows(unique_rows, seed), unique_rows, size)
    if len(chosen) != size:
        raise ValueError(f"cannot select {size} rows with distinct sentence hashes and polygons")
    return chosen


def _validate_sample_parameters(size: int, seed: int) -> None:
    _validate_sample_size(size)
    _validate_sample_seed(seed)


def _validate_sample_size(size: int) -> None:
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")


def _validate_sample_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _observed_names(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in rows:
        code = _required_string(row, "eunis_code")
        name = _required_string(row, "eunis_name")
        existing = names.get(code)
        if existing is not None and existing != name:
            raise ValueError(f"inconsistent EUNIS name for {code}")
        names[code] = name
    return names


def _validate_candidate_name(code: str, observed_name: str, details: Mapping[str, str]) -> str:
    name = _required_string(details, "name")
    if name != observed_name:
        raise ValueError(f"EUNIS name mismatch for {code}")
    return name


def _validate_candidate_description(code: str, details: Mapping[str, str]) -> str:
    description = _required_string(details, "description")
    if not description.strip():
        raise ValueError(f"description must be non-empty for {code}")
    return description


def _validate_candidate_source_hash(code: str, source_hash: str) -> None:
    if len(source_hash) != 64 or any(ch not in "0123456789abcdef" for ch in source_hash):
        raise ValueError(f"source_sha256 must be a lowercase SHA-256 for {code}")


def _candidate_record(code: str, observed_name: str, details: Mapping[str, str]) -> dict[str, str]:
    name = _validate_candidate_name(code, observed_name, details)
    description = _validate_candidate_description(code, details)
    fields = (
        "classification_release",
        "classification_source",
        "source_sha256",
        "license",
    )
    record = {field: _required_string(details, field) for field in fields}
    source_hash = record["source_sha256"]
    _validate_candidate_source_hash(code, source_hash)
    return {
        "eunis_code": code,
        "eunis_name": name,
        "eunis_description": description,
        "candidate_text": f"{name}\n{description}",
        **record,
    }


def build_candidate_labels(
    rows: Iterable[Mapping[str, Any]], taxonomy: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    """Build sorted candidate labels from pinned polygon names and taxonomy descriptions."""
    names = _observed_names(rows)
    if set(names) != set(taxonomy):
        raise ValueError("taxonomy codes do not match observed EUNIS codes")
    return [_candidate_record(code, names[code], taxonomy[code]) for code in sorted(names)]
