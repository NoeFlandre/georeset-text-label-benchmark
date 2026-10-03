"""Metrics and breakdowns for the finite candidate-label pilot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any


def _validate_lengths(
    gold: Sequence[str], top1: Sequence[str], top5: Sequence[Sequence[str]]
) -> None:
    if not gold or len(gold) != len(top1) or len(gold) != len(top5):
        raise ValueError("gold, top1, and top5 lengths must agree and be non-empty")


def _require_known(values: Sequence[str], known: set[str], role: str) -> None:
    for value in values:
        if value not in known:
            raise ValueError(f"unknown {role} code: {value}")


def _validate_candidate_codes(candidate_codes: Sequence[str]) -> set[str]:
    if not candidate_codes or len(set(candidate_codes)) != len(candidate_codes):
        raise ValueError("candidate codes must be non-empty and unique")
    return set(candidate_codes)


def _validate_top5(first: str, predictions: Sequence[str], known: set[str]) -> None:
    if not predictions or len(predictions) > 5 or len(set(predictions)) != len(predictions):
        raise ValueError("top5 must contain one to five unique candidate codes")
    if predictions[0] != first:
        raise ValueError("top1 prediction must be first in top5")
    _require_known(predictions, known, "prediction")


def _validate_predictions(
    gold: Sequence[str],
    top1: Sequence[str],
    top5: Sequence[Sequence[str]],
    candidate_codes: Sequence[str],
) -> None:
    _validate_lengths(gold, top1, top5)
    known = _validate_candidate_codes(candidate_codes)
    _require_known(gold, known, "gold")
    _require_known(top1, known, "prediction")
    for first, predictions in zip(top1, top5, strict=True):
        _validate_top5(first, predictions, known)


def _true_positives(gold: Sequence[str], top1: Sequence[str | None]) -> Counter[str]:
    return Counter(
        actual for actual, predicted in zip(gold, top1, strict=True) if actual == predicted
    )


def _macro_f1(
    gold: Sequence[str], top1: Sequence[str | None], candidate_codes: Sequence[str]
) -> float:
    supports = Counter(gold)
    predicted = Counter(top1)
    true_positives = _true_positives(gold, top1)
    scores = [
        2.0 * true_positives[code] / (supports[code] + predicted[code])
        if supports[code] + predicted[code]
        else 0.0
        for code in candidate_codes
    ]
    return sum(scores) / len(candidate_codes)


def _validate_single_label_predictions(
    gold: Sequence[str], predictions: Sequence[str | None], candidate_codes: Sequence[str]
) -> set[str]:
    if not gold or len(gold) != len(predictions):
        raise ValueError("gold and single-label prediction lengths must agree and be non-empty")
    known = _validate_candidate_codes(candidate_codes)
    _require_known(gold, known, "gold")
    _require_known([code for code in predictions if code is not None], known, "prediction")
    return known


def single_label_report(
    gold: Sequence[str],
    predictions: Sequence[str | None],
    candidate_codes: Sequence[str],
) -> dict[str, Any]:
    """Report direct-label metrics; invalid outputs count as wrong and reduce coverage."""
    _validate_single_label_predictions(gold, predictions, candidate_codes)
    prediction_count = sum(code is not None for code in predictions)
    correct = sum(actual == predicted for actual, predicted in zip(gold, predictions, strict=True))
    return {
        "sample_count": len(gold),
        "candidate_class_count": len(candidate_codes),
        "prediction_count": prediction_count,
        "invalid_output_count": len(gold) - prediction_count,
        "coverage": prediction_count / len(gold),
        "top1_accuracy": correct / len(gold),
        "macro_f1_all_candidates": _macro_f1(gold, predictions, candidate_codes),
        "macro_f1_definition": (
            "Unweighted mean of per-code F1 over all candidate codes; invalid outputs count as "
            "misses, and codes with no gold support and no predictions contribute zero."
        ),
    }


def single_label_class_breakdown(
    gold: Sequence[str],
    predictions: Sequence[str | None],
    candidate_codes: Sequence[str],
) -> list[dict[str, Any]]:
    """Report support and direct-label precision/recall for every candidate code."""
    _validate_single_label_predictions(gold, predictions, candidate_codes)
    support = Counter(gold)
    predicted = Counter(code for code in predictions if code is not None)
    true_positives = _true_positives(gold, predictions)
    return [
        _single_label_class_row(code, support, predicted, true_positives)
        for code in candidate_codes
    ]


def _single_label_class_row(
    code: str,
    support: Counter[str],
    predicted: Counter[str],
    true_positives: Counter[str],
) -> dict[str, Any]:
    count = support[code]
    prediction_count = predicted[code]
    hits = true_positives[code]
    return {
        "eunis_code": code,
        "support": count,
        "prediction_count": prediction_count,
        "top1_correct": hits,
        "top1_precision": hits / prediction_count if prediction_count else None,
        "top1_recall": hits / count if count else None,
    }


def single_label_group_breakdown(
    groups: Sequence[str | None],
    gold: Sequence[str],
    predictions: Sequence[str | None],
) -> list[dict[str, Any]]:
    """Report top-1 accuracy and valid-output coverage for each observed group."""
    if len(groups) != len(gold) or len(predictions) != len(gold) or not gold:
        raise ValueError("groups, gold, and predictions lengths must agree and be non-empty")
    names = _normalize_groups(groups)
    indices_by_group = _indices_by_group(names)
    return [
        _single_label_group_row(name, indices, gold, predictions)
        for name, indices in indices_by_group.items()
    ]


def _single_label_group_row(
    name: str,
    indices: Sequence[int],
    gold: Sequence[str],
    predictions: Sequence[str | None],
) -> dict[str, Any]:
    correct = sum(gold[index] == predictions[index] for index in indices)
    valid = sum(predictions[index] is not None for index in indices)
    return {
        "group": name,
        "sample_count": len(indices),
        "prediction_count": valid,
        "coverage": valid / len(indices),
        "top1_accuracy": correct / len(indices),
    }


def _accuracy_counts(
    gold: Sequence[str], top1: Sequence[str], top5: Sequence[Sequence[str]]
) -> tuple[int, int]:
    correct = sum(actual == predicted for actual, predicted in zip(gold, top1, strict=True))
    top5_hits = sum(actual in predictions for actual, predictions in zip(gold, top5, strict=True))
    return correct, top5_hits


def classification_report(
    gold: Sequence[str],
    top1: Sequence[str],
    top5: Sequence[Sequence[str]],
    candidate_codes: Sequence[str],
) -> dict[str, Any]:
    """Compute top-1, top-5, and all-candidate macro-F1 metrics."""
    _validate_predictions(gold, top1, top5, candidate_codes)
    correct, top5_hits = _accuracy_counts(gold, top1, top5)
    return {
        "sample_count": len(gold),
        "candidate_class_count": len(candidate_codes),
        "top1_accuracy": correct / len(gold),
        "top5_accuracy": top5_hits / len(gold),
        "macro_f1_all_candidates": _macro_f1(gold, top1, candidate_codes),
        "macro_f1_definition": (
            "Unweighted mean of per-code F1 over all candidate codes; codes with no gold "
            "support and no predictions contribute zero."
        ),
    }


def class_breakdown(
    gold: Sequence[str],
    top1: Sequence[str],
    top5: Sequence[Sequence[str]],
    candidate_codes: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute support and hit rates for every candidate class."""
    _validate_predictions(gold, top1, top5, candidate_codes)
    support = Counter(gold)
    top1_hits = _true_positives(gold, top1)
    top5_hits = Counter(
        actual for actual, predictions in zip(gold, top5, strict=True) if actual in predictions
    )
    return [_class_metrics(code, support, top1_hits, top5_hits) for code in candidate_codes]


def _class_metrics(
    code: str,
    support: Counter[str],
    top1_hits: Counter[str],
    top5_hits: Counter[str],
) -> dict[str, Any]:
    count = support[code]
    accuracy = top1_hits[code] / count if count else None
    recall = top5_hits[code] / count if count else None
    return {
        "eunis_code": code,
        "support": count,
        "top1_correct": top1_hits[code],
        "top1_accuracy": accuracy,
        "top5_hits": top5_hits[code],
        "top5_recall": recall,
    }


def _group_row(
    name: str,
    indices: Sequence[int],
    gold: Sequence[str],
    top1: Sequence[str],
    top5: Sequence[Sequence[str]],
) -> dict[str, Any]:
    actual = [gold[index] for index in indices]
    predicted = [top1[index] for index in indices]
    predictions = [top5[index] for index in indices]
    correct, hits = _accuracy_counts(actual, predicted, predictions)
    return {
        "group": name,
        "sample_count": len(indices),
        "top1_accuracy": correct / len(indices),
        "top5_accuracy": hits / len(indices),
    }


def _normalize_groups(groups: Sequence[str | None]) -> list[str]:
    return [group if group is not None else "missing" for group in groups]


def _indices_by_group(names: Sequence[str]) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        indices.setdefault(name, []).append(index)
    return dict(sorted(indices.items()))


def group_breakdown(
    groups: Sequence[str | None],
    gold: Sequence[str],
    top1: Sequence[str],
    top5: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    """Compute top-1 and top-5 hit rates for each observed group."""
    _validate_lengths(gold, top1, top5)
    if len(groups) != len(gold):
        raise ValueError("groups, gold, top1, and top5 lengths must agree")
    names = _normalize_groups(groups)
    indices_by_group = _indices_by_group(names)
    return [
        _group_row(name, indices, gold, top1, top5) for name, indices in indices_by_group.items()
    ]
