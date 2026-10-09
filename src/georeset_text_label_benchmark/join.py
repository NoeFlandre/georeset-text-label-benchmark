"""Occurrence-safe, source-agnostic text-to-polygon join primitives."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from georeset_text_label_benchmark.errors import (
    CardinalityError,
    DataValidationError,
    ProvenanceError,
)
from georeset_text_label_benchmark.models import AuditCounts, GlobalKeys, PartitionResult

DescriptionKey = tuple[str, str]
PolygonKey = tuple[str, int]
SENTENCE_DECISIONS = frozenset({"yes", "no", "failed"})
ALL_DECISIONS = SENTENCE_DECISIONS | {"skipped_unsplit"}
LABELLED_DECISIONS = frozenset({"yes", "no"})
BBOX_FIELDS = ("bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y")


def process_partition(
    labels: Iterable[Mapping[str, Any]],
    descriptions: Iterable[Mapping[str, Any]],
    polygons: Iterable[Mapping[str, Any]],
    *,
    input_revision: str,
    partition_name: str,
    expected_eunis_source_version: str,
    global_keys: GlobalKeys | None = None,
    labelled: bool = False,
) -> PartitionResult:
    """Join one bounded source partition and return eligible rows plus audit counts.

    Default rows are yes sentences with an EUNIS assignment. With ``labelled`` set, yes and
    no sentences with an assignment are kept, each with its decision and polygon bbox.
    """
    seen = global_keys if global_keys is not None else GlobalKeys()
    audit = AuditCounts()
    polygon_index = _index_polygons(
        polygons, seen, audit, partition_name, expected_eunis_source_version
    )
    description_index = _index_descriptions(
        descriptions, polygon_index, seen, audit, partition_name
    )
    rows = _join_labels(
        labels,
        description_index,
        polygon_index,
        seen,
        audit,
        input_revision,
        partition_name,
        labelled,
    )
    return PartitionResult(rows, audit)


def _required_text(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{context}: {field} must be a non-empty string")
    return value


def _required_integer(row: Mapping[str, Any], field: str, context: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataValidationError(f"{context}: {field} must be an integer")
    return value


def _polygon_key(row: Mapping[str, Any], context: str) -> PolygonKey:
    osm_type = _required_text(row, "osm_type", context)
    osm_id = _required_integer(row, "osm_id", context)
    if osm_id <= 0:
        raise DataValidationError(f"{context}: osm_id must be positive")
    return osm_type, osm_id


def _description_key(row: Mapping[str, Any], context: str) -> DescriptionKey:
    identity = _required_text(row, "description_identity", context)
    tag_key = _required_text(row, "tag_key", context)
    return identity, tag_key


def _validate_eunis_assignment(
    row: Mapping[str, Any], context: str, expected_eunis_source_version: str
) -> str | None:
    code = row.get("eunis_code")
    name = row.get("eunis_name")
    overlap = row.get("eunis_overlap_percentage")
    version = row.get("eunis_source_version")
    if code is None:
        if any(value is not None for value in (name, overlap, version)):
            raise DataValidationError(
                f"{context}: missing EUNIS code has non-null companion fields"
            )
        return None
    _validate_assigned_eunis(code, name, overlap, version, context)
    if version != expected_eunis_source_version:
        raise ProvenanceError(
            f"{context}: eunis_source_version {version!r} differs from pinned "
            f"EUNIS reference version {expected_eunis_source_version!r}"
        )
    return code


def _validate_assigned_eunis(
    code: Any,
    name: Any,
    overlap: Any,
    version: Any,
    context: str,
) -> None:
    _validate_eunis_text(code, "eunis_code", context)
    _validate_eunis_text(name, "eunis_name", context)
    _validate_eunis_text(version, "eunis_source_version", context)
    _validate_overlap_percentage(overlap, context)


def _validate_eunis_text(value: Any, field: str, context: str) -> None:
    if not isinstance(value, str):
        raise DataValidationError(f"{context}: {field} must be a non-empty string")
    if not value.strip():
        raise DataValidationError(f"{context}: {field} must be a non-empty string")


def _validate_overlap_percentage(overlap: Any, context: str) -> None:
    if type(overlap) not in (int, float):
        raise DataValidationError(f"{context}: assigned EUNIS code needs a numeric overlap")
    if not math.isfinite(overlap):
        raise DataValidationError(
            f"{context}: EUNIS overlap percentage must be finite and in [0, 100]"
        )
    if not 0 <= overlap <= 100:
        raise DataValidationError(
            f"{context}: EUNIS overlap percentage must be finite and in [0, 100]"
        )


def _index_polygons(
    rows: Iterable[Mapping[str, Any]],
    seen: GlobalKeys,
    audit: AuditCounts,
    partition_name: str,
    expected_eunis_source_version: str,
) -> dict[PolygonKey, Mapping[str, Any]]:
    polygons: dict[PolygonKey, Mapping[str, Any]] = {}
    for row in rows:
        context = f"{partition_name} polygon"
        _required_text(row, "source_pbf", context)
        key = _polygon_key(row, context)
        if key in polygons or key in seen.polygon_keys:
            raise CardinalityError(f"{context}: duplicate global polygon key {key}")
        _validate_eunis_assignment(row, context, expected_eunis_source_version)
        polygons[key] = row
        seen.polygon_keys.add(key)
        audit.polygon_rows += 1
        if row.get("eunis_code") is not None:
            audit.assigned_polygon_rows += 1
    return polygons


def _validate_description_sentences(row: Mapping[str, Any], context: str) -> Sequence[str]:
    sentences = _validate_sentence_list(row.get("sentences"), context)
    _validate_sentence_count(row.get("sentence_count"), sentences, context)
    _validate_sentence_values(sentences, context)
    return sentences


def _validate_sentence_list(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence):
        raise DataValidationError(f"{context}: sentences must be a list of strings")
    if isinstance(value, (str, bytes)):
        raise DataValidationError(f"{context}: sentences must be a list of strings")
    return value


def _validate_sentence_count(count: Any, sentences: Sequence[Any], context: str) -> None:
    if not isinstance(count, int):
        raise DataValidationError(f"{context}: sentence_count does not match sentences length")
    if isinstance(count, bool):
        raise DataValidationError(f"{context}: sentence_count does not match sentences length")
    if count != len(sentences):
        raise DataValidationError(f"{context}: sentence_count does not match sentences length")


def _validate_sentence_values(sentences: Sequence[Any], context: str) -> None:
    if any(not isinstance(sentence, str) for sentence in sentences):
        raise DataValidationError(f"{context}: every sentence must be a string")


def _index_descriptions(
    rows: Iterable[Mapping[str, Any]],
    polygons: Mapping[PolygonKey, Mapping[str, Any]],
    seen: GlobalKeys,
    audit: AuditCounts,
    partition_name: str,
) -> dict[DescriptionKey, Mapping[str, Any]]:
    descriptions: dict[DescriptionKey, Mapping[str, Any]] = {}
    for row in rows:
        context = f"{partition_name} description-tag"
        key = _description_key(row, context)
        _require_unique_description_key(key, descriptions, seen, context)
        polygon_key = _polygon_key(row, context)
        polygon = _require_polygon(polygon_key, polygons, context)
        _validate_source_provenance(row, polygon, context)
        _validate_description_sentences(row, context)
        _validate_language_code(row.get("language_code"), context)
        descriptions[key] = row
        seen.description_keys.add(key)
        seen.described_polygon_keys.add(polygon_key)
        audit.description_tag_rows += 1
    return descriptions


def _require_unique_description_key(
    key: DescriptionKey,
    descriptions: Mapping[DescriptionKey, Mapping[str, Any]],
    seen: GlobalKeys,
    context: str,
) -> None:
    if key in descriptions or key in seen.description_keys:
        raise CardinalityError(f"{context}: duplicate global description key {key}")


def _require_polygon(
    key: PolygonKey, polygons: Mapping[PolygonKey, Mapping[str, Any]], context: str
) -> Mapping[str, Any]:
    polygon = polygons.get(key)
    if polygon is None:
        raise CardinalityError(f"{context}: unmatched EUNIS polygon key {key}")
    return polygon


def _validate_language_code(language_code: Any, context: str) -> None:
    if language_code is None:
        return
    if not isinstance(language_code, str):
        raise DataValidationError(f"{context}: language_code must be a string or null")


def _validate_source_provenance(
    description: Mapping[str, Any], polygon: Mapping[str, Any], context: str
) -> None:
    description_source = _required_text(description, "source_pbf", context)
    if description_source != polygon["source_pbf"]:
        raise ProvenanceError(
            f"{context}: description source_pbf {description_source!r} differs from polygon "
            f"source_pbf {polygon['source_pbf']!r}"
        )


def _validate_label(
    row: Mapping[str, Any], input_revision: str, partition_name: str
) -> tuple[DescriptionKey, int, str, str]:
    context = f"{partition_name} label"
    key = _description_key(row, context)
    index = _required_integer(row, "sentence_index", context)
    decision = _required_text(row, "decision", context)
    _validate_decision(decision, context)
    _validate_revision(row.get("input_revision"), input_revision, context)
    text_hash = _validate_text_hash(row.get("text_sha256"), context)
    return key, index, decision, text_hash


def _validate_decision(decision: str, context: str) -> None:
    if decision not in ALL_DECISIONS:
        raise DataValidationError(f"{context}: unrecognized decision {decision!r}")


def _validate_revision(value: Any, input_revision: str, context: str) -> None:
    observed_revision = _required_text({"input_revision": value}, "input_revision", context)
    if observed_revision != input_revision:
        raise ProvenanceError(f"{context}: input_revision does not match pinned Description source")


def _validate_text_hash(value: Any, context: str) -> str:
    text_hash = _required_text({"text_sha256": value}, "text_sha256", context)
    if len(text_hash) != 64:
        raise DataValidationError(f"{context}: text_sha256 must be a 64-character SHA-256 digest")
    try:
        valid_digest = len(bytes.fromhex(text_hash)) == 32
    except ValueError:
        valid_digest = False
    if not valid_digest:
        raise DataValidationError(f"{context}: text_sha256 must contain 64 hexadecimal characters")
    return text_hash


def _sentence_for_label(
    description: Mapping[str, Any], index: int, decision: str, context: str
) -> str | None:
    sentences = _validate_description_sentences(description, context)
    if decision == "skipped_unsplit":
        _validate_unsplit_sentinel(sentences, index, context)
        return None
    return _sentence_at_index(sentences, index, context)


def _validate_unsplit_sentinel(sentences: Sequence[str], index: int, context: str) -> None:
    if len(sentences) != 0 or index != 0:
        raise DataValidationError(f"{context}: skipped_unsplit must be an empty index-0 sentinel")


def _sentence_at_index(sentences: Sequence[str], index: int, context: str) -> str:
    if index < 0 or index >= len(sentences):
        raise DataValidationError(f"{context}: sentence_index {index} is outside the sentence list")
    return sentences[index]


def _verify_sentence_and_language(
    label: Mapping[str, Any],
    description: Mapping[str, Any],
    sentence: str | None,
    text_hash: str,
    context: str,
) -> None:
    _validate_language_pair(label.get("language"), description.get("language_code"), context)
    _verify_sentence_hash(sentence, text_hash, context)


def _validate_language_pair(language: Any, language_code: Any, context: str) -> None:
    if language is None:
        return
    if not isinstance(language, str):
        raise DataValidationError(f"{context}: label language must be a string or null")
    if language_code is None:
        return
    if language != language_code:
        raise ProvenanceError(f"{context}: label language differs from detected language_code")


def _verify_sentence_hash(sentence: str | None, text_hash: str, context: str) -> None:
    if sentence is None:
        return
    actual_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
    if actual_hash != text_hash:
        raise DataValidationError(f"{context}: text_sha256 does not match exact sentence bytes")


def _overlap_record(
    description: Mapping[str, Any],
    polygon: Mapping[str, Any],
    sentence: str,
    text_hash: str,
    index: int,
) -> dict[str, Any]:
    return {
        "source_pbf": description["source_pbf"],
        "osm_type": description["osm_type"],
        "osm_id": description["osm_id"],
        "description_identity": description["description_identity"],
        "tag_key": description["tag_key"],
        "sentence_index": index,
        "sentence": sentence,
        "text_sha256": text_hash,
        "language_code": description.get("language_code"),
        "eunis_code": polygon["eunis_code"],
        "eunis_name": polygon["eunis_name"],
        "eunis_overlap_percentage": polygon["eunis_overlap_percentage"],
        "eunis_source_version": polygon["eunis_source_version"],
    }


def _record_label(
    decision: str,
    description: Mapping[str, Any],
    polygon: Mapping[str, Any],
    sentence: str | None,
    text_hash: str,
    index: int,
    audit: AuditCounts,
    labelled: bool = False,
) -> dict[str, Any] | None:
    code = polygon.get("eunis_code")
    _count_label(audit, decision, _assignment_name(code))
    if sentence is None:
        audit.skipped_unsplit_rows += 1
        return None
    audit.sentence_rows += 1
    audit.unique_sentence_hashes.add(text_hash)
    if decision not in _retained_decisions(labelled):
        return None
    if code is None:
        _count_unassigned(audit, decision)
        return None
    if decision == "yes":
        _count_overlap(audit, description, code)
    return _retained_record(labelled, decision, description, polygon, sentence, text_hash, index)


def _count_label(audit: AuditCounts, decision: str, assignment: str) -> None:
    audit.label_rows += 1
    audit.decisions[decision] += 1
    audit.decision_assignment[(decision, assignment)] += 1


def _retained_decisions(labelled: bool) -> frozenset[str]:
    return LABELLED_DECISIONS if labelled else frozenset({"yes"})


def _count_unassigned(audit: AuditCounts, decision: str) -> None:
    if decision == "yes":
        audit.yes_without_eunis_rows += 1


def _retained_record(
    labelled: bool,
    decision: str,
    description: Mapping[str, Any],
    polygon: Mapping[str, Any],
    sentence: str,
    text_hash: str,
    index: int,
) -> dict[str, Any]:
    if labelled:
        return _labelled_record(decision, description, polygon, sentence, text_hash, index)
    return _overlap_record(description, polygon, sentence, text_hash, index)


def _count_overlap(audit: AuditCounts, description: Mapping[str, Any], code: str) -> None:
    audit.retained_rows += 1
    audit.overlap_by_eunis[code] += 1
    audit.overlap_by_tag[description["tag_key"]] += 1
    audit.overlap_by_language[description.get("language_code") or "unknown"] += 1


def _labelled_record(
    decision: str,
    description: Mapping[str, Any],
    polygon: Mapping[str, Any],
    sentence: str,
    text_hash: str,
    index: int,
) -> dict[str, Any]:
    record = _overlap_record(description, polygon, sentence, text_hash, index)
    record["decision"] = decision
    for field in BBOX_FIELDS:
        record[field] = _finite_number(polygon, field)
    return record


def _finite_number(polygon: Mapping[str, Any], field: str) -> float:
    value = polygon.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise DataValidationError(f"polygon {field} must be a finite number")
    return float(value)


def _assignment_name(code: Any) -> str:
    return "assigned" if code is not None else "missing"


def _join_labels(
    rows: Iterable[Mapping[str, Any]],
    descriptions: Mapping[DescriptionKey, Mapping[str, Any]],
    polygons: Mapping[PolygonKey, Mapping[str, Any]],
    seen: GlobalKeys,
    audit: AuditCounts,
    input_revision: str,
    partition_name: str,
    labelled: bool = False,
) -> list[dict[str, Any]]:
    observed_by_description: Counter[DescriptionKey] = Counter()
    overlaps: list[dict[str, Any]] = []
    for label in rows:
        context = f"{partition_name} label"
        key, index, decision, text_hash = _validate_label(label, input_revision, partition_name)
        sentence_key = (*key, index)
        if sentence_key in seen.sentence_keys:
            raise CardinalityError(f"{context}: duplicate sentence occurrence key {sentence_key}")
        description = descriptions.get(key)
        if description is None:
            raise CardinalityError(f"{context}: unmatched description key {key}")
        sentence = _sentence_for_label(description, index, decision, context)
        _verify_sentence_and_language(label, description, sentence, text_hash, context)
        observed_by_description[key] += 1
        seen.sentence_keys.add(sentence_key)
        polygon = _polygon_for_description(description, polygons, context)
        row = _record_label(
            decision, description, polygon, sentence, text_hash, index, audit, labelled
        )
        if row is not None:
            overlaps.append(row)
    _verify_label_coverage(descriptions, observed_by_description, partition_name)
    return overlaps


def _polygon_for_description(
    description: Mapping[str, Any],
    polygons: Mapping[PolygonKey, Mapping[str, Any]],
    context: str,
) -> Mapping[str, Any]:
    polygon_key = _polygon_key(description, context)
    polygon = polygons.get(polygon_key)
    if polygon is None:
        raise CardinalityError(f"{context}: description polygon key {polygon_key} is unmatched")
    return polygon


def _verify_label_coverage(
    descriptions: Mapping[DescriptionKey, Mapping[str, Any]],
    observed: Counter[DescriptionKey],
    partition_name: str,
) -> None:
    for key, description in descriptions.items():
        expected = description["sentence_count"] or 1
        actual = observed[key]
        if actual != expected:
            raise CardinalityError(
                f"{partition_name}: description {key} has {actual} labels for {expected} "
                "sentence positions"
            )
