"""Version-pinned retrospective scoring at official EUNIS ancestor levels."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from georeset_text_label_benchmark.pilot.metrics import (
    single_label_group_breakdown,
    single_label_report,
)

UNRESOLVED_PREDICTION = "__EUNIS_UNRESOLVED_PREDICTION__"
UNMAPPED_GOLD = "__EUNIS_UNMAPPED_GOLD__"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class _Node:
    parent_code: str | None
    level: int


@dataclass(frozen=True)
class _Classification:
    release: str
    source: str
    source_sha256: str
    levels: tuple[int, ...]
    nodes: Mapping[str, _Node]


def hierarchical_report(
    *,
    groups: Sequence[str],
    gold_codes: Sequence[str],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_versions: Sequence[str],
    candidate_codes: Sequence[str],
    candidate_sources: Mapping[str, Mapping[str, str]],
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    """Score exact official ancestors at common levels, preserving the original leaf score."""
    size = _validate_aligned_rows(
        groups, gold_codes, predicted_codes, parse_statuses, source_versions
    )
    candidates = _validate_candidates(candidate_codes)
    classifications = _parse_classifications(taxonomy)
    source_by_code = _validate_candidate_sources(candidates, candidate_sources, classifications)
    _validate_row_values(gold_codes, source_versions, candidates)
    _reject_reserved_codes(classifications)

    levels = sorted({level for item in classifications.values() for level in item.levels})
    universes = _label_universes(candidates, source_by_code, classifications, levels)
    rows = _score_levels(
        levels,
        universes,
        groups,
        gold_codes,
        predicted_codes,
        parse_statuses,
        source_by_code,
        classifications,
        size,
    )
    return {
        "protocol_version": "eunis-hierarchy-v1",
        "classifications": _classification_provenance(classifications),
        "row_source_versions": sorted(set(source_versions)),
        "definitions": _metric_definitions(),
        "original_leaf": {
            "overall": single_label_report(gold_codes, predicted_codes, candidate_codes),
            "by_decision": single_label_group_breakdown(groups, gold_codes, predicted_codes),
        },
        "rows": rows,
    }


def _validate_aligned_rows(
    groups: Sequence[str],
    gold_codes: Sequence[str],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_versions: Sequence[str],
) -> int:
    lengths = {
        len(groups),
        len(gold_codes),
        len(predicted_codes),
        len(parse_statuses),
        len(source_versions),
    }
    if len(lengths) != 1 or not groups:
        raise ValueError("hierarchical scoring inputs must have equal non-zero lengths")
    if any(group not in {"yes", "no"} for group in groups):
        raise ValueError("group must be 'yes' or 'no'")
    return len(groups)


def _validate_candidates(candidate_codes: Sequence[str]) -> set[str]:
    if not candidate_codes:
        raise ValueError("candidate codes must be non-empty and unique")
    if len(set(candidate_codes)) != len(candidate_codes):
        raise ValueError("candidate codes must be non-empty and unique")
    if any(_invalid_candidate_code(code) for code in candidate_codes):
        raise ValueError("candidate codes must be non-empty strings without edge whitespace")
    return set(candidate_codes)


def _invalid_candidate_code(code: str) -> bool:
    return not isinstance(code, str) or not code or code != code.strip()


def _parse_classifications(payload: Mapping[str, Any]) -> dict[str, _Classification]:
    return _unique_classifications(_classification_records(payload))


def _classification_records(payload: Mapping[str, Any]) -> list[Any]:
    if payload.get("format_version") != 1:
        raise ValueError("taxonomy format_version must be 1")
    records = payload.get("classifications")
    if not isinstance(records, list) or not records:
        raise ValueError("taxonomy classifications must be a non-empty list")
    return records


def _unique_classifications(records: Sequence[Any]) -> dict[str, _Classification]:
    result: dict[str, _Classification] = {}
    for record in records:
        classification = _parse_classification(record)
        if classification.source_sha256 in result:
            raise ValueError("taxonomy classification source hashes must be unique")
        result[classification.source_sha256] = classification
    return result


def _parse_classification(value: Any) -> _Classification:
    if not isinstance(value, Mapping):
        raise ValueError("taxonomy classifications must be objects")
    release = _required_text(value, "classification_release")
    source = _required_text(value, "classification_source")
    source_hash = _required_text(value, "source_sha256")
    if not _SHA256_RE.fullmatch(source_hash):
        raise ValueError("taxonomy source_sha256 must be a lowercase SHA-256")
    levels = _parse_levels(value.get("levels"))
    nodes = _parse_nodes(value.get("nodes"), levels)
    _validate_parents(nodes, levels)
    return _Classification(release, source, source_hash, levels, nodes)


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"taxonomy {field} must be a non-empty string without edge whitespace")
    return value


def _parse_levels(value: Any) -> tuple[int, ...]:
    if not _is_integer_level_list(value):
        raise ValueError("taxonomy levels must be a non-empty ordered integer list")
    levels = tuple(value)
    _validate_positive_levels(levels)
    _validate_level_order(levels)
    return levels


def _is_integer_level_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(type(item) is int for item in value)


def _validate_positive_levels(levels: Sequence[int]) -> None:
    if any(level < 1 for level in levels):
        raise ValueError("taxonomy levels must be positive integers")


def _validate_level_order(levels: Sequence[int]) -> None:
    if tuple(sorted(set(levels))) != tuple(levels):
        raise ValueError("taxonomy levels must be unique and ascending")


def _parse_nodes(value: Any, levels: Sequence[int]) -> dict[str, _Node]:
    if not isinstance(value, list) or not value:
        raise ValueError("taxonomy nodes must be a non-empty list")
    nodes: dict[str, _Node] = {}
    known_levels = set(levels)
    for entry in value:
        code, node = _parse_node(entry, known_levels)
        if code in nodes:
            raise ValueError(f"duplicate taxonomy node code: {code}")
        nodes[code] = node
    return nodes


def _parse_node(entry: Any, known_levels: set[int]) -> tuple[str, _Node]:
    if not isinstance(entry, Mapping):
        raise ValueError("taxonomy nodes must be objects")
    code = entry.get("code")
    parent = entry.get("parent_code")
    level = entry.get("level")
    code, parent, level = _validate_node_fields(code, parent, level, known_levels)
    return code, _Node(parent, level)


def _validate_node_fields(
    code: Any, parent: Any, level: Any, known_levels: set[int]
) -> tuple[str, str | None, int]:
    valid_code = _validate_node_code(code)
    valid_parent = _validate_parent_code(parent, valid_code)
    valid_level = _validate_node_level(level, valid_code, known_levels)
    return valid_code, valid_parent, valid_level


def _validate_node_code(code: Any) -> str:
    if not isinstance(code, str) or not code or code != code.strip():
        raise ValueError("taxonomy node code must be a non-empty string")
    return code


def _validate_parent_code(parent: Any, code: str) -> str | None:
    if parent is not None and (
        not isinstance(parent, str) or not parent or parent != parent.strip()
    ):
        raise ValueError(f"taxonomy parent code must be null or a non-empty string: {code}")
    return parent


def _validate_node_level(level: Any, code: str, known_levels: set[int]) -> int:
    if type(level) is not int or level not in known_levels:
        raise ValueError(f"taxonomy node has an unknown level: {code}")
    return level


def _validate_parents(nodes: Mapping[str, _Node], levels: Sequence[int]) -> None:
    for code, node in nodes.items():
        parent_code = node.parent_code
        if parent_code is None:
            continue
        _validate_parent(code, parent_code, node.level, nodes, levels)


def _validate_parent(
    code: str,
    parent_code: str,
    child_level: int,
    nodes: Mapping[str, _Node],
    levels: Sequence[int],
) -> None:
    if parent_code not in nodes:
        raise ValueError(f"taxonomy parent is missing for {code}: {parent_code}")
    parent_level = nodes[parent_code].level
    if parent_level >= child_level or parent_level not in levels:
        raise ValueError(f"taxonomy parent level must precede child level: {code}")


def _validate_candidate_sources(
    candidates: set[str],
    candidate_sources: Mapping[str, Mapping[str, str]],
    classifications: Mapping[str, _Classification],
) -> dict[str, str]:
    if set(candidate_sources) != candidates:
        raise ValueError("candidate source metadata must match the candidate codes")
    source_by_code = {
        code: _candidate_source_hash(code, candidate_sources[code], classifications)
        for code in sorted(candidates)
    }
    if set(source_by_code.values()) != set(classifications):
        raise ValueError("taxonomy classifications must match candidate taxonomy sources")
    return source_by_code


def _candidate_source_hash(
    code: str,
    metadata: Mapping[str, str],
    classifications: Mapping[str, _Classification],
) -> str:
    source_hash = metadata.get("source_sha256")
    classification = classifications.get(source_hash or "")
    if classification is None:
        raise ValueError(f"candidate taxonomy source does not match mapping: {code}")
    if any(
        metadata.get(field) != expected
        for field, expected in (
            ("classification_release", classification.release),
            ("classification_source", classification.source),
            ("source_sha256", classification.source_sha256),
        )
    ):
        raise ValueError(f"candidate taxonomy source does not match mapping: {code}")
    return classification.source_sha256


def _validate_row_values(
    gold_codes: Sequence[str], source_versions: Sequence[str], candidates: set[str]
) -> None:
    _validate_gold_codes(gold_codes, candidates)
    _validate_source_versions(source_versions)


def _validate_gold_codes(gold_codes: Sequence[str], candidates: set[str]) -> None:
    if any(code not in candidates for code in gold_codes):
        raise ValueError("gold code is outside the candidate vocabulary")


def _validate_source_versions(source_versions: Sequence[str]) -> None:
    if any(not version or version != version.strip() for version in source_versions):
        raise ValueError("prediction rows need a non-empty EUNIS source version")


def _reject_reserved_codes(classifications: Mapping[str, _Classification]) -> None:
    for classification in classifications.values():
        if UNRESOLVED_PREDICTION in classification.nodes or UNMAPPED_GOLD in classification.nodes:
            raise ValueError("taxonomy contains a reserved evaluation sentinel code")


def _classification_provenance(
    classifications: Mapping[str, _Classification],
) -> list[dict[str, Any]]:
    return [
        {
            "classification_release": item.release,
            "classification_source": item.source,
            "source_sha256": item.source_sha256,
            "levels": list(item.levels),
        }
        for _, item in sorted(classifications.items())
    ]


def _label_universes(
    candidates: set[str],
    source_by_code: Mapping[str, str],
    classifications: Mapping[str, _Classification],
    levels: Sequence[int],
) -> dict[int, list[str]]:
    found = {level: set() for level in levels}
    for code in candidates:
        classification = classifications[source_by_code[code]]
        _add_candidate_ancestors(code, source_by_code[code], classification, levels, found)
    return {level: sorted(tokens) for level, tokens in found.items()}


def _add_candidate_ancestors(
    code: str,
    source_hash: str,
    classification: _Classification,
    levels: Sequence[int],
    found: dict[int, set[str]],
) -> None:
    for level in levels:
        ancestor = _ancestor_at_level(code, level, classification)
        if ancestor is not None:
            found[level].add(_label_token(source_hash, ancestor))


def _label_token(source_hash: str, code: str) -> str:
    return f"{source_hash}:{code}"


def _ancestor_at_level(code: str, level: int, classification: _Classification) -> str | None:
    current = code
    while current in classification.nodes:
        node = classification.nodes[current]
        if node.level == level:
            return current
        if node.level < level or node.parent_code is None:
            return None
        current = node.parent_code
    return None


def _reference_level(
    code: str, source_hash: str, classifications: Mapping[str, _Classification]
) -> int | None:
    node = classifications[source_hash].nodes.get(code)
    return node.level if node is not None else None


def _score_levels(
    levels: Sequence[int],
    universes: Mapping[int, Sequence[str]],
    groups: Sequence[str],
    gold_codes: Sequence[str],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
    classifications: Mapping[str, _Classification],
    size: int,
) -> list[dict[str, Any]]:
    results = []
    for level in levels:
        for group in ("no", "overall", "yes"):
            indices = _group_indices(groups, group, size)
            eligible, excluded = _level_eligibility(
                indices, level, gold_codes, source_by_code, classifications
            )
            results.append(
                _score_level_group(
                    level,
                    group,
                    indices,
                    eligible,
                    excluded,
                    gold_codes,
                    predicted_codes,
                    parse_statuses,
                    source_by_code,
                    classifications,
                    universes[level],
                )
            )
    return results


def _group_indices(groups: Sequence[str], group: str, size: int) -> list[int]:
    if group == "overall":
        return list(range(size))
    return [index for index, value in enumerate(groups) if value == group]


def _level_eligibility(
    indices: Sequence[int],
    level: int,
    gold_codes: Sequence[str],
    source_by_code: Mapping[str, str],
    classifications: Mapping[str, _Classification],
) -> tuple[list[int], int]:
    if level <= 3:
        return list(indices), 0
    eligible = []
    excluded = 0
    for index in indices:
        reference_level = _reference_level(
            gold_codes[index], source_by_code[gold_codes[index]], classifications
        )
        if reference_level is None or reference_level >= level:
            eligible.append(index)
        else:
            excluded += 1
    return eligible, excluded


def _score_level_group(
    level: int,
    group: str,
    source_indices: Sequence[int],
    indices: Sequence[int],
    excluded: int,
    gold_codes: Sequence[str],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
    classifications: Mapping[str, _Classification],
    label_universe: Sequence[str],
) -> dict[str, Any]:
    gold, predicted = _map_group_codes(
        indices,
        level,
        gold_codes,
        predicted_codes,
        parse_statuses,
        source_by_code,
        classifications,
    )
    valid_count, failed_count, truncated_count = _prediction_status_counts(
        indices, predicted_codes, parse_statuses, source_by_code
    )
    unmapped_gold, unmapped_prediction, unmapped = _unmapped_counts(
        indices, gold, predicted, predicted_codes, parse_statuses, source_by_code
    )
    correct = _correct_count(gold, predicted)
    valid_mapped = _valid_mapped_count(
        indices, gold, predicted, predicted_codes, parse_statuses, source_by_code
    )
    return {
        "level": level,
        "group": group,
        "sample_n": len(source_indices),
        "eligible_n": len(indices),
        "n": len(indices),
        "excluded_shallower_reference_count": excluded,
        "valid_count": valid_count,
        "failed_count": failed_count,
        "truncated_count": truncated_count,
        "unmapped_count": unmapped,
        "unmapped_gold_count": unmapped_gold,
        "unmapped_prediction_count": unmapped_prediction,
        "correct_count": correct,
        "accuracy": _ratio(correct, len(indices)),
        "valid_mapped_count": valid_mapped,
        "valid_only_accuracy": _ratio(correct, valid_mapped),
        "macro_f1": _macro_f1(gold, predicted, label_universe) if indices else None,
        "multiclass_mcc": _multiclass_mcc(gold, predicted, label_universe) if indices else None,
        "label_universe": list(label_universe),
    }


def _map_group_codes(
    indices: Sequence[int],
    level: int,
    gold_codes: Sequence[str],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
    classifications: Mapping[str, _Classification],
) -> tuple[list[str | None], list[str | None]]:
    gold: list[str | None] = []
    predicted: list[str | None] = []
    for index in indices:
        gold.append(
            _mapped_code(
                gold_codes[index], gold_codes[index], level, source_by_code, classifications
            )
        )
        predicted.append(
            _map_prediction_code(
                index,
                level,
                gold_codes,
                predicted_codes,
                parse_statuses,
                source_by_code,
                classifications,
            )
        )
    return gold, predicted


def _map_prediction_code(
    index: int,
    level: int,
    gold_codes: Sequence[str],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
    classifications: Mapping[str, _Classification],
) -> str | None:
    code = predicted_codes[index]
    if parse_statuses[index] != "valid" or code not in source_by_code:
        return None
    return _mapped_code(gold_codes[index], code, level, source_by_code, classifications)


def _prediction_status_counts(
    indices: Sequence[int],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
) -> tuple[int, int, int]:
    valid = sum(
        _valid_prediction(index, predicted_codes, parse_statuses, source_by_code)
        for index in indices
    )
    truncated = sum(parse_statuses[index] == "truncated" for index in indices)
    return valid, len(indices) - valid - truncated, truncated


def _valid_prediction(
    index: int,
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
) -> bool:
    return parse_statuses[index] == "valid" and predicted_codes[index] in source_by_code


def _unmapped_counts(
    indices: Sequence[int],
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
) -> tuple[int, int, int]:
    unmapped_gold = sum(code is None for code in gold)
    unmapped_prediction = sum(
        _unmapped_prediction(index, guess, predicted_codes, parse_statuses, source_by_code)
        for index, guess in zip(indices, predicted, strict=True)
    )
    unmapped = sum(
        _unmapped_row(index, actual, guess, parse_statuses)
        for index, actual, guess in zip(indices, gold, predicted, strict=True)
    )
    return unmapped_gold, unmapped_prediction, unmapped


def _unmapped_prediction(
    index: int,
    guess: str | None,
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
) -> bool:
    return (
        parse_statuses[index] == "valid"
        and predicted_codes[index] in source_by_code
        and guess is None
    )


def _unmapped_row(
    index: int,
    actual: str | None,
    guess: str | None,
    parse_statuses: Sequence[str],
) -> bool:
    return actual is None or (parse_statuses[index] == "valid" and guess is None)


def _correct_count(gold: Sequence[str | None], predicted: Sequence[str | None]) -> int:
    return sum(_is_correct(actual, guess) for actual, guess in zip(gold, predicted, strict=True))


def _is_correct(actual: str | None, guess: str | None) -> bool:
    return actual is not None and guess is not None and actual == guess


def _valid_mapped_count(
    indices: Sequence[int],
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
) -> int:
    return sum(
        _has_valid_mapped_prediction(
            index,
            actual,
            guess,
            predicted_codes,
            parse_statuses,
            source_by_code,
        )
        for index, actual, guess in zip(indices, gold, predicted, strict=True)
    )


def _has_valid_mapped_prediction(
    index: int,
    actual: str | None,
    guess: str | None,
    predicted_codes: Sequence[str | None],
    parse_statuses: Sequence[str],
    source_by_code: Mapping[str, str],
) -> bool:
    return (
        _valid_prediction(index, predicted_codes, parse_statuses, source_by_code)
        and actual is not None
        and guess is not None
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mapped_code(
    reference_code: str,
    code: str | None,
    level: int,
    source_by_code: Mapping[str, str],
    classifications: Mapping[str, _Classification],
) -> str | None:
    if code is None:
        return None
    reference_source = source_by_code[reference_code]
    if source_by_code.get(code) != reference_source:
        return None
    ancestor = _ancestor_at_level(code, level, classifications[reference_source])
    return _label_token(reference_source, ancestor) if ancestor is not None else None


def _macro_f1(
    gold: Sequence[str | None], predicted: Sequence[str | None], labels: Sequence[str]
) -> float:
    supports = _label_counts(gold)
    predictions = _label_counts(predicted)
    true_positives = _true_positive_counts(gold, predicted)
    scores = [_class_f1(label, supports, predictions, true_positives) for label in labels]
    return _mean(scores)


def _label_counts(codes: Sequence[str | None]) -> Counter[str]:
    return Counter(code for code in codes if code is not None)


def _true_positive_counts(
    gold: Sequence[str | None], predicted: Sequence[str | None]
) -> Counter[str]:
    return Counter(
        actual
        for actual, guess in zip(gold, predicted, strict=True)
        if actual is not None and actual == guess
    )


def _class_f1(
    label: str,
    supports: Counter[str],
    predictions: Counter[str],
    true_positives: Counter[str],
) -> float:
    denominator = supports[label] + predictions[label]
    return 2 * true_positives[label] / denominator if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _multiclass_mcc(
    gold: Sequence[str | None], predicted: Sequence[str | None], labels: Sequence[str]
) -> float:
    true_labels = [code if code is not None else UNMAPPED_GOLD for code in gold]
    predicted_labels = [code if code is not None else UNRESOLVED_PREDICTION for code in predicted]
    return _mcc_from_labels(
        true_labels,
        predicted_labels,
        [*labels, UNMAPPED_GOLD, UNRESOLVED_PREDICTION],
    )


def _mcc_from_labels(gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> float:
    true_counts = Counter(gold)
    predicted_counts = Counter(predicted)
    count = len(gold)
    if not count:
        return 0.0
    correct = sum(actual == guess for actual, guess in zip(gold, predicted, strict=True))
    numerator = _mcc_numerator(count, correct, true_counts, predicted_counts, labels)
    denominator = _mcc_denominator(count, true_counts, predicted_counts, labels)
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _mcc_numerator(
    count: int,
    correct: int,
    true_counts: Counter[str],
    predicted_counts: Counter[str],
    labels: Sequence[str],
) -> int:
    class_overlap = sum(predicted_counts[label] * true_counts[label] for label in labels)
    return count * correct - class_overlap


def _mcc_denominator(
    count: int,
    true_counts: Counter[str],
    predicted_counts: Counter[str],
    labels: Sequence[str],
) -> float:
    predicted_variance = _count_variance(count, predicted_counts, labels)
    true_variance = _count_variance(count, true_counts, labels)
    product = predicted_variance * true_variance
    return product**0.5 if product > 0 else 0.0


def _count_variance(count: int, counts: Counter[str], labels: Sequence[str]) -> int:
    squared_counts = sum(counts[label] ** 2 for label in labels)
    return count * count - squared_counts


def _metric_definitions() -> dict[str, str]:
    return {
        "level_1_to_3_scope": (
            "Levels 1-3 include every sample row in each yes/no group. A reference that lacks a "
            "target-level ancestor remains in n and is counted unmapped."
        ),
        "level_4_scope": (
            "Level 4 includes references whose exact source taxonomy places the gold code at level "
            "4 or deeper, plus unmapped gold references retained for accountability. Shallower "
            "references are explicitly excluded and counted in excluded_shallower_reference_count."
        ),
        "label_universe": (
            "For each level, all distinct official ancestor labels reached from the complete frozen "
            "candidate vocabulary at that exact level, qualified by source SHA-256. The same fixed "
            "universe is used for yes, no, and overall rows."
        ),
        "accuracy": (
            "Correct mapped ancestor predictions divided by eligible n. Failed, truncated, and "
            "unmapped outcomes are incorrect; every eligible row stays in the denominator. All "
            "model metrics are null when eligible n is zero."
        ),
        "macro_f1": (
            "Unweighted mean over the level's complete label universe. Failed, truncated, and "
            "unmapped predictions add no predicted class and remain false negatives for mapped gold "
            "classes; unmapped gold rows cannot be true positives. Null when eligible n is 0."
        ),
        "multiclass_mcc": (
            "Generalized multiclass MCC over official labels plus distinct unmapped-gold and "
            "unresolved-prediction sentinels. The sentinels never match each other. For a non-empty "
            "eligible set, returns 0 when the denominator is zero; it is null when eligible n is 0."
        ),
        "valid_only_accuracy": (
            "Correct mapped ancestor predictions divided by eligible rows with a parser-valid "
            "predicted code and mapped gold and prediction ancestors; supplementary to accuracy."
        ),
        "counts": (
            "valid_count, failed_count, and truncated_count partition eligible n by output status. "
            "unmapped_count is the number of eligible rows with an unmapped gold or valid predicted "
            "code at that level and may overlap those status counts."
        ),
        "original_leaf": (
            "The original 158-candidate direct-label report and yes/no breakdown are repeated "
            "without coarsening, over all records and including invalid outputs in the denominator."
        ),
    }
