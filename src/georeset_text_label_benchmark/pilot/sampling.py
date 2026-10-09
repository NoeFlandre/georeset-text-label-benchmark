"""Candidate-label construction for the pinned EUNIS vocabulary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


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
