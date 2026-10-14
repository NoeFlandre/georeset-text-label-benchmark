"""Retrospective EUNIS hierarchy scoring preserves leaf results and taxonomic versions."""

from __future__ import annotations

import copy
import csv
import json
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from georeset_text_label_benchmark.pilot.hierarchy import (
    UNMAPPED_GOLD,
    UNRESOLVED_PREDICTION,
    _correct_count,
    _has_valid_mapped_prediction,
    _macro_f1,
    _mcc_denominator,
    _mcc_from_labels,
    _multiclass_mcc,
    _parse_classifications,
    _reference_level,
    _true_positive_counts,
    _unmapped_counts,
    _valid_mapped_count,
    hierarchical_report,
)
from georeset_text_label_benchmark.pilot.metrics import single_label_report


def _metadata(classification: str, source: str, source_hash: str) -> dict[str, str]:
    return {
        "classification_release": classification,
        "classification_source": source,
        "source_sha256": source_hash,
    }


def _taxonomy() -> dict[str, Any]:
    return {
        "format_version": 1,
        "classifications": [
            {
                **_metadata("EUNIS-test-terrestrial", "https://example.test/terrestrial", "a" * 64),
                "levels": [1, 2, 3],
                "nodes": [
                    {"code": "A0", "parent_code": None, "level": 1},
                    {"code": "A1", "parent_code": "A0", "level": 2},
                    {"code": "X", "parent_code": "A0", "level": 2},
                    {"code": "A11", "parent_code": "A1", "level": 3},
                    {"code": "A12", "parent_code": "A1", "level": 3},
                ],
            },
            {
                **_metadata("EUNIS-test-marine", "https://example.test/marine", "b" * 64),
                "levels": [1, 2, 3, 4],
                "nodes": [
                    {"code": "B0", "parent_code": None, "level": 1},
                    {"code": "B1", "parent_code": "B0", "level": 2},
                    {"code": "B2", "parent_code": "B1", "level": 3},
                    {"code": "B11", "parent_code": "B2", "level": 4},
                ],
            },
        ],
    }


def _candidate_sources() -> dict[str, dict[str, str]]:
    terrestrial = _metadata("EUNIS-test-terrestrial", "https://example.test/terrestrial", "a" * 64)
    marine = _metadata("EUNIS-test-marine", "https://example.test/marine", "b" * 64)
    return {"A11": terrestrial, "A12": terrestrial, "B11": marine, "X": terrestrial}


def _report() -> dict[str, Any]:
    return hierarchical_report(
        groups=["yes", "yes", "no", "no"],
        gold_codes=["A11", "A12", "B11", "X"],
        predicted_codes=["A12", None, "B11", None],
        parse_statuses=["valid", "truncated", "valid", "invalid"],
        source_versions=["EEA EUNIS habitat probability maps v1 2021"] * 4,
        candidate_codes=["A11", "A12", "B11", "X"],
        candidate_sources=_candidate_sources(),
        taxonomy=_taxonomy(),
    )


def _evaluate(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "groups": ["yes"],
        "gold_codes": ["A11"],
        "predicted_codes": ["A11"],
        "parse_statuses": ["valid"],
        "source_versions": ["EEA map v1"],
        "candidate_codes": ["A11", "A12", "B11", "X"],
        "candidate_sources": _candidate_sources(),
        "taxonomy": _taxonomy(),
    }
    arguments.update(overrides)
    return hierarchical_report(**arguments)


@contextmanager
def _raises_exact(expected_type: type[Exception], message: str) -> Iterator[None]:
    with pytest.raises(expected_type) as caught:
        yield
    assert str(caught.value) == message


_TAXONOMY_MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "format_version": lambda taxonomy: taxonomy.update(format_version=2),
    "empty_classifications": lambda taxonomy: taxonomy.update(classifications=[]),
    "non_object_classification": lambda taxonomy: taxonomy.update(classifications=[None]),
    "duplicate_source": lambda taxonomy: taxonomy["classifications"][1].update(
        source_sha256="a" * 64
    ),
    "invalid_hash": lambda taxonomy: taxonomy["classifications"][0].update(source_sha256="G" * 64),
    "empty_release": lambda taxonomy: taxonomy["classifications"][0].update(
        classification_release=" "
    ),
    "non_text_release": lambda taxonomy: taxonomy["classifications"][0].update(
        classification_release=7
    ),
    "empty_source": lambda taxonomy: taxonomy["classifications"][0].update(
        classification_source=" "
    ),
    "non_text_source": lambda taxonomy: taxonomy["classifications"][0].update(
        classification_source=7
    ),
    "levels_not_list": lambda taxonomy: taxonomy["classifications"][0].update(levels="1,2,3"),
    "empty_levels": lambda taxonomy: taxonomy["classifications"][0].update(levels=[]),
    "bool_level": lambda taxonomy: taxonomy["classifications"][0].update(levels=[True]),
    "nonpositive_level": lambda taxonomy: taxonomy["classifications"][0].update(levels=[0]),
    "descending_levels": lambda taxonomy: taxonomy["classifications"][0].update(levels=[2, 1]),
    "empty_nodes": lambda taxonomy: taxonomy["classifications"][0].update(nodes=[]),
    "non_object_node": lambda taxonomy: taxonomy["classifications"][0].update(nodes=[None]),
    "empty_code": lambda taxonomy: taxonomy["classifications"][0]["nodes"][0].update(code=" A0"),
    "non_text_code": lambda taxonomy: taxonomy["classifications"][0]["nodes"][0].update(code=7),
    "bad_parent": lambda taxonomy: taxonomy["classifications"][0]["nodes"][1].update(parent_code=4),
    "empty_parent": lambda taxonomy: taxonomy["classifications"][0]["nodes"][1].update(
        parent_code=""
    ),
    "unknown_node_level": lambda taxonomy: taxonomy["classifications"][0]["nodes"][0].update(
        level=True
    ),
    "duplicate_node": lambda taxonomy: taxonomy["classifications"][0]["nodes"].append(
        copy.deepcopy(taxonomy["classifications"][0]["nodes"][0])
    ),
    "missing_parent": lambda taxonomy: taxonomy["classifications"][0]["nodes"][1].update(
        parent_code="missing"
    ),
    "unordered_parent": lambda taxonomy: taxonomy["classifications"][0]["nodes"][1].update(level=1),
    "reserved_code": lambda taxonomy: taxonomy["classifications"][0]["nodes"].append(
        {"code": UNMAPPED_GOLD, "parent_code": None, "level": 1}
    ),
}


def test_hierarchical_report_scores_levels_and_explicit_level_four_eligibility() -> None:
    report = _report()
    assert set(report) == {
        "protocol_version",
        "classifications",
        "row_source_versions",
        "definitions",
        "original_leaf",
        "rows",
    }
    assert report["protocol_version"] == "eunis-hierarchy-v1"
    assert report["classifications"] == [
        {
            "classification_release": "EUNIS-test-terrestrial",
            "classification_source": "https://example.test/terrestrial",
            "source_sha256": "a" * 64,
            "levels": [1, 2, 3],
        },
        {
            "classification_release": "EUNIS-test-marine",
            "classification_source": "https://example.test/marine",
            "source_sha256": "b" * 64,
            "levels": [1, 2, 3, 4],
        },
    ]
    assert report["row_source_versions"] == ["EEA EUNIS habitat probability maps v1 2021"]
    assert report["definitions"] == _metric_definitions_expected()
    rows = report["rows"]
    assert isinstance(rows, list)
    by_key = {(row["level"], row["group"]): row for row in rows}
    assert list(by_key) == [
        (1, "no"),
        (1, "overall"),
        (1, "yes"),
        (2, "no"),
        (2, "overall"),
        (2, "yes"),
        (3, "no"),
        (3, "overall"),
        (3, "yes"),
        (4, "no"),
        (4, "overall"),
        (4, "yes"),
    ]

    level1 = by_key[(1, "overall")]
    assert level1["n"] == level1["eligible_n"] == 4
    assert level1["correct_count"] == 2
    assert level1["accuracy"] == pytest.approx(0.5)
    assert (level1["valid_count"], level1["failed_count"], level1["truncated_count"]) == (2, 1, 1)
    assert level1["unmapped_count"] == 0
    assert level1["macro_f1"] == pytest.approx(0.75)
    assert level1["multiclass_mcc"] == pytest.approx(4 / (60**0.5))
    assert level1["label_universe"] == [f"{'a' * 64}:A0", f"{'b' * 64}:B0"]
    assert by_key[(1, "yes")]["accuracy"] == pytest.approx(0.5)
    assert by_key[(1, "no")]["accuracy"] == pytest.approx(0.5)
    assert all(
        by_key[(level, group)]["excluded_shallower_reference_count"] == 0
        for level in (1, 2, 3)
        for group in ("yes", "no", "overall")
    )

    level3 = by_key[(3, "overall")]
    assert level3["n"] == 4
    assert level3["accuracy"] == pytest.approx(0.25)
    assert level3["unmapped_gold_count"] == 1
    assert level3["unmapped_prediction_count"] == 0
    assert level3["unmapped_count"] == 1
    assert level3["valid_mapped_count"] == 2
    assert level3["valid_only_accuracy"] == pytest.approx(0.5)
    assert level3["label_universe"] == [
        f"{'a' * 64}:A11",
        f"{'a' * 64}:A12",
        f"{'b' * 64}:B2",
    ]

    level4 = by_key[(4, "overall")]
    assert level4["sample_n"] == 4
    assert level4["eligible_n"] == level4["n"] == 1
    assert level4["excluded_shallower_reference_count"] == 3
    assert level4["accuracy"] == 1.0
    assert level4["label_universe"] == [f"{'b' * 64}:B11"]
    assert by_key[(4, "yes")]["eligible_n"] == 0
    assert by_key[(4, "yes")]["excluded_shallower_reference_count"] == 2
    assert by_key[(4, "yes")]["accuracy"] is None
    assert by_key[(4, "yes")]["valid_only_accuracy"] is None
    assert by_key[(4, "yes")]["macro_f1"] is None
    assert by_key[(4, "yes")]["multiclass_mcc"] is None
    assert by_key[(4, "no")]["eligible_n"] == 1
    assert by_key[(4, "no")]["excluded_shallower_reference_count"] == 1
    assert by_key[(4, "no")]["accuracy"] == 1.0
    assert by_key[(4, "overall")]["accuracy"] == 1.0


def _metric_definitions_expected() -> dict[str, str]:
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


def test_hierarchical_scoring_keeps_official_classification_sources_separate() -> None:
    report = hierarchical_report(
        groups=["yes", "no"],
        gold_codes=["A11", "B11"],
        predicted_codes=["A11", "A11"],
        parse_statuses=["valid", "valid"],
        source_versions=["EEA map v1"] * 2,
        candidate_codes=["A11", "B11"],
        candidate_sources={
            "A11": _candidate_sources()["A11"],
            "B11": _candidate_sources()["B11"],
        },
        taxonomy=_taxonomy(),
    )
    level1 = next(row for row in report["rows"] if row["level"] == 1 and row["group"] == "overall")

    assert level1["n"] == 2
    assert level1["accuracy"] == 0.5
    assert level1["unmapped_prediction_count"] == 1
    assert level1["label_universe"] == [f"{'a' * 64}:A0", f"{'b' * 64}:B0"]


def test_fine_metrics_remain_unchanged_by_hierarchical_scoring() -> None:
    before = single_label_report(["A", "B", "A"], ["A", None, "B"], ["A", "B", "C"])
    report = _report()
    after = single_label_report(["A", "B", "A"], ["A", None, "B"], ["A", "B", "C"])

    assert after == before
    assert after["top1_accuracy"] == pytest.approx(1 / 3)
    assert after["macro_f1_all_candidates"] == pytest.approx(2 / 9)
    assert report["original_leaf"]["overall"] == single_label_report(
        ["A11", "A12", "B11", "X"],
        ["A12", None, "B11", None],
        ["A11", "A12", "B11", "X"],
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("duplicate_levels", "taxonomy levels must be unique and ascending"),
        ("source_mismatch", "candidate taxonomy source does not match mapping: A11"),
    ],
)
def test_hierarchical_report_rejects_bad_or_mismatched_taxonomy(change: str, message: str) -> None:
    taxonomy = _taxonomy()
    sources = _candidate_sources()
    if change == "duplicate_levels":
        taxonomy["classifications"][0]["levels"] = [1, 1]  # type: ignore[index]
    else:
        taxonomy["classifications"][0]["classification_release"] = "different release"  # type: ignore[index]
    with _raises_exact(ValueError, message):
        hierarchical_report(
            groups=["yes"],
            gold_codes=["A11"],
            predicted_codes=["A11"],
            parse_statuses=["valid"],
            source_versions=["EEA map v1"],
            candidate_codes=["A11", "A12", "B11", "X"],
            candidate_sources=sources,
            taxonomy=taxonomy,
        )


def test_hierarchical_report_rejects_non_yes_no_group_membership() -> None:
    with _raises_exact(ValueError, "group must be 'yes' or 'no'"):
        hierarchical_report(
            groups=["maybe"],
            gold_codes=["A11"],
            predicted_codes=["A11"],
            parse_statuses=["valid"],
            source_versions=["EEA map v1"],
            candidate_codes=["A11", "A12", "B11", "X"],
            candidate_sources=_candidate_sources(),
            taxonomy=_taxonomy(),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("format_version", "taxonomy format_version must be 1"),
        ("empty_classifications", "taxonomy classifications must be a non-empty list"),
        ("non_object_classification", "taxonomy classifications must be objects"),
        ("duplicate_source", "taxonomy classification source hashes must be unique"),
        ("invalid_hash", "taxonomy source_sha256 must be a lowercase SHA-256"),
        (
            "empty_release",
            "taxonomy classification_release must be a non-empty string without edge whitespace",
        ),
        (
            "non_text_release",
            "taxonomy classification_release must be a non-empty string without edge whitespace",
        ),
        (
            "empty_source",
            "taxonomy classification_source must be a non-empty string without edge whitespace",
        ),
        (
            "non_text_source",
            "taxonomy classification_source must be a non-empty string without edge whitespace",
        ),
        ("levels_not_list", "taxonomy levels must be a non-empty ordered integer list"),
        ("empty_levels", "taxonomy levels must be a non-empty ordered integer list"),
        ("bool_level", "taxonomy levels must be a non-empty ordered integer list"),
        ("nonpositive_level", "taxonomy levels must be positive integers"),
        ("descending_levels", "taxonomy levels must be unique and ascending"),
        ("empty_nodes", "taxonomy nodes must be a non-empty list"),
        ("non_object_node", "taxonomy nodes must be objects"),
        ("empty_code", "taxonomy node code must be a non-empty string"),
        ("non_text_code", "taxonomy node code must be a non-empty string"),
        ("bad_parent", "taxonomy parent code must be null or a non-empty string: A1"),
        ("empty_parent", "taxonomy parent code must be null or a non-empty string: A1"),
        ("unknown_node_level", "taxonomy node has an unknown level: A0"),
        ("duplicate_node", "duplicate taxonomy node code: A0"),
        ("missing_parent", "taxonomy parent is missing for A1: missing"),
        ("unordered_parent", "taxonomy parent level must precede child level: A1"),
        ("reserved_code", "taxonomy contains a reserved evaluation sentinel code"),
    ],
)
def test_hierarchical_report_rejects_malformed_taxonomy(change: str, message: str) -> None:
    taxonomy = copy.deepcopy(_taxonomy())
    _TAXONOMY_MUTATIONS[change](taxonomy)

    with _raises_exact(ValueError, message):
        _evaluate(taxonomy=taxonomy)


@pytest.mark.parametrize(
    ("candidate_codes", "message"),
    [
        ([], "candidate codes must be non-empty and unique"),
        (["A11", "A11"], "candidate codes must be non-empty and unique"),
        ([""], "candidate codes must be non-empty strings without edge whitespace"),
        ([" A11"], "candidate codes must be non-empty strings without edge whitespace"),
        ([1], "candidate codes must be non-empty strings without edge whitespace"),
    ],
)
def test_hierarchical_report_rejects_bad_candidate_codes(
    candidate_codes: list[Any], message: str
) -> None:
    with _raises_exact(ValueError, message):
        _evaluate(candidate_codes=candidate_codes)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("candidate_keys", "candidate source metadata must match the candidate codes"),
        ("unknown_source", "candidate taxonomy source does not match mapping: A11"),
        ("unused_classification", "taxonomy classifications must match candidate taxonomy sources"),
    ],
)
def test_hierarchical_report_rejects_bad_candidate_source_metadata(
    change: str, message: str
) -> None:
    sources = _candidate_sources()
    candidates = ["A11", "A12", "B11", "X"]
    if change == "candidate_keys":
        sources.pop("X")
    elif change == "unknown_source":
        sources["A11"]["source_sha256"] = "c" * 64
    else:
        candidates = ["A11"]
        sources = {"A11": sources["A11"]}

    with _raises_exact(ValueError, message):
        _evaluate(candidate_codes=candidates, candidate_sources=sources)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gold_codes", ["UNKNOWN"], "gold code is outside the candidate vocabulary"),
        ("source_versions", [" "], "prediction rows need a non-empty EUNIS source version"),
    ],
)
def test_hierarchical_report_rejects_bad_reference_metadata(
    field: str, value: list[str], message: str
) -> None:
    with _raises_exact(ValueError, message):
        _evaluate(**{field: value})


def test_hierarchical_report_rejects_empty_or_misaligned_row_vectors() -> None:
    with _raises_exact(ValueError, "hierarchical scoring inputs must have equal non-zero lengths"):
        _evaluate(
            groups=[], gold_codes=[], predicted_codes=[], parse_statuses=[], source_versions=[]
        )
    with _raises_exact(ValueError, "hierarchical scoring inputs must have equal non-zero lengths"):
        _evaluate(predicted_codes=[])


def test_invalid_predictions_remain_failures_and_metric_edges_are_defined() -> None:
    report = _evaluate(
        groups=["yes", "no"],
        gold_codes=["A11", "A12"],
        predicted_codes=[None, None],
        parse_statuses=["invalid", "invalid"],
        source_versions=["EEA map v1"] * 2,
    )
    overall = next(row for row in report["rows"] if row["level"] == 1 and row["group"] == "overall")
    assert (overall["n"], overall["valid_count"], overall["failed_count"]) == (2, 0, 2)
    assert overall["truncated_count"] == 0
    assert overall["accuracy"] == 0
    assert overall["macro_f1"] == 0
    assert overall["multiclass_mcc"] == 0
    assert _macro_f1(["A"], ["A"], ["A", "UNUSED"]) == 0.5
    assert _macro_f1([], [], []) == 0
    assert _multiclass_mcc(["A", "B"], [None, None], ["A", "B"]) == 0
    assert _multiclass_mcc(["A"], ["A"], ["A"]) == 0
    assert _mcc_from_labels([], [], []) == 0
    assert UNRESOLVED_PREDICTION != UNMAPPED_GOLD


def test_invalid_status_cannot_become_a_valid_prediction_or_a_mapped_answer() -> None:
    report = _evaluate(predicted_codes=["A11"], parse_statuses=["invalid"])
    overall = next(row for row in report["rows"] if row["level"] == 3 and row["group"] == "overall")
    assert (overall["valid_count"], overall["failed_count"]) == (0, 1)
    assert (overall["correct_count"], overall["valid_mapped_count"], overall["accuracy"]) == (
        0,
        0,
        0,
    )
    assert not _has_valid_mapped_prediction(
        0,
        "A11",
        "A11",
        ["A11"],
        ["invalid"],
        {"A11": "a" * 64},
    )


def test_valid_but_unmappable_prediction_is_counted_as_unmapped() -> None:
    report = _evaluate(predicted_codes=["X"], parse_statuses=["valid"])
    overall = next(row for row in report["rows"] if row["level"] == 3 and row["group"] == "overall")
    assert (overall["valid_count"], overall["unmapped_prediction_count"]) == (1, 1)
    assert (overall["unmapped_count"], overall["correct_count"]) == (1, 0)


def test_unmapped_gold_and_unresolved_prediction_use_distinct_mcc_labels() -> None:
    assert _multiclass_mcc([None, "A"], [None, "A"], ["A"]) == pytest.approx(0.5)
    assert _mcc_from_labels(["A", "B"], ["A", "B"], ["A", "B"]) == 1.0
    assert _mcc_from_labels(["A"], ["A"], ["A"]) == 0.0


def test_mcc_denominator_handles_zero_and_positive_class_variance() -> None:
    assert _mcc_denominator(1, Counter(A=1), Counter(A=1), ["A"]) == 0.0
    assert _mcc_denominator(2, Counter(A=1, B=1), Counter(A=1, B=1), ["A", "B"]) == 2.0


def test_unmapped_reference_has_no_level_for_level_four_eligibility() -> None:
    classifications = _parse_classifications(_taxonomy())
    assert _reference_level("missing", "a" * 64, classifications) is None
    assert _reference_level("A11", "a" * 64, classifications) == 3


def test_pairwise_metric_helpers_reject_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match=r"^zip\(\) argument"):
        _correct_count(["A"], [])
    with pytest.raises(ValueError, match=r"^zip\(\) argument"):
        _true_positive_counts(["A"], [])
    with pytest.raises(ValueError, match=r"^zip\(\) argument"):
        _mcc_from_labels(["A"], [], ["A"])
    with pytest.raises(ValueError, match=r"^zip\(\) argument"):
        _unmapped_counts([0], [None], [], [None], ["invalid"], {})
    with pytest.raises(ValueError, match=r"^zip\(\) argument"):
        _unmapped_counts([0], [], [None], [None], ["invalid"], {})
    with pytest.raises(ValueError, match=r"^zip\(\) argument"):
        _valid_mapped_count([0], [None], [], [None], ["invalid"], {})


def test_pinned_eunis_mapping_matches_workbooks_and_historical_sample_eligibility() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = json.loads((root / "artifacts/pilot-100-seed42/frozen_sample.json").read_text())
    taxonomy = json.loads((root / "artifacts/eunis-hierarchy-v1/taxonomy_mapping.json").read_text())
    with (root / "pilot_data/eunis_candidate_labels.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        candidates = list(csv.DictReader(stream))
    candidate_codes = [row["eunis_code"] for row in candidates]
    candidate_sources = {
        row["eunis_code"]: {
            key: row[key]
            for key in ("classification_release", "classification_source", "source_sha256")
        }
        for row in candidates
    }
    rows = sample["selected_rows"]
    report = hierarchical_report(
        groups=[row["decision"] for row in rows],
        gold_codes=[row["eunis_code"] for row in rows],
        predicted_codes=[None] * len(rows),
        parse_statuses=["truncated"] * len(rows),
        source_versions=[row["eunis_source_version"] for row in rows],
        candidate_codes=candidate_codes,
        candidate_sources=candidate_sources,
        taxonomy=taxonomy,
    )
    by_level = {row["level"]: row for row in report["rows"] if row["group"] == "overall"}
    verification = taxonomy["verification"]

    assert taxonomy["candidate_csv_sha256"] == sample["candidate_labels"]["sha256"]
    assert verification["ancestor_counts_by_level"] == {"1": 7, "2": 28, "3": 152, "4": 9}
    assert (
        verification["sample_coverage"]["sample_ids_sha256"]
        == sample["selection"]["sample_ids_sha256"]
    )
    assert verification["sample_coverage"]["reference_eligibility_by_level"]["4"] == {
        "yes_eligible_n": 0,
        "no_eligible_n": 0,
        "eligible_n": 0,
        "excluded_shallower_reference_count": 100,
    }
    assert [by_level[level]["n"] for level in (1, 2, 3)] == [100, 100, 100]
    assert [len(by_level[level]["label_universe"]) for level in (1, 2, 3, 4)] == [7, 28, 152, 9]
    assert by_level[4]["n"] == 0
    assert by_level[4]["accuracy"] is None
    assert by_level[4]["valid_only_accuracy"] is None
    assert by_level[4]["macro_f1"] is None
    assert by_level[4]["multiclass_mcc"] is None
    assert all(
        row["macro_f1"] is None and row["multiclass_mcc"] is None
        for row in report["rows"]
        if row["level"] == 4
    )
    assert by_level[4]["excluded_shallower_reference_count"] == 100
    assert Counter(row["decision"] for row in rows) == {"yes": 50, "no": 50}
    assert report["original_leaf"]["overall"]["sample_count"] == 100
    for level in (1, 2, 3):
        assert (
            next(row for row in report["rows"] if row["level"] == level and row["group"] == "yes")[
                "n"
            ]
            == 50
        )
        assert (
            next(row for row in report["rows"] if row["level"] == level and row["group"] == "no")[
                "n"
            ]
            == 50
        )
