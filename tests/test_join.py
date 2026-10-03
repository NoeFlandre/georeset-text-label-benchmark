"""Contract tests for occurrence-preserving Description joins."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from georeset_text_label_benchmark.errors import (
    CardinalityError,
    DataValidationError,
    ProvenanceError,
)
from georeset_text_label_benchmark.join import _polygon_for_description, process_partition
from georeset_text_label_benchmark.models import GlobalKeys

INPUT_REVISION = "b4706eb315c66a8a57135289e7612dca8e03caf8"
SOURCE_PBF = "tuvalu-latest.osm.pbf"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _polygon(osm_id: int, *, code: str | None = "T11") -> dict[str, Any]:
    return {
        "source_pbf": SOURCE_PBF,
        "osm_type": "way",
        "osm_id": osm_id,
        "eunis_code": code,
        "eunis_name": "Temperate riparian forest" if code else None,
        "eunis_overlap_percentage": 72.5 if code else None,
        "eunis_source_version": "EEA EUNIS habitat probability maps v1 2021" if code else None,
    }


def _description(
    identity: str,
    osm_id: int,
    sentences: list[str],
    *,
    tag_key: str = "description",
    language: str | None = "eng",
) -> dict[str, Any]:
    return {
        "description_identity": identity,
        "tag_key": tag_key,
        "source_pbf": SOURCE_PBF,
        "osm_type": "way",
        "osm_id": osm_id,
        "language_code": language,
        "sentence_count": len(sentences),
        "sentences": sentences,
    }


def _label(
    identity: str,
    index: int,
    sentence: str,
    decision: str,
    *,
    tag_key: str = "description",
    language: str | None = "eng",
) -> dict[str, Any]:
    return {
        "description_identity": identity,
        "tag_key": tag_key,
        "sentence_index": index,
        "text_sha256": _hash(sentence),
        "decision": decision,
        "language": language,
        "input_revision": INPUT_REVISION,
    }


def _process(
    labels: list[dict[str, Any]],
    descriptions: list[dict[str, Any]],
    polygons: list[dict[str, Any]],
    *,
    global_keys: GlobalKeys | None = None,
) -> Any:
    return process_partition(
        labels,
        descriptions,
        polygons,
        input_revision=INPUT_REVISION,
        partition_name="tuvalu-latest.parquet",
        global_keys=global_keys,
    )


def test_retains_yes_sentence_with_assigned_eunis_and_preserves_exact_text() -> None:
    sentence = "  meadow\nedge "
    result = _process(
        [_label("desc-1", 0, sentence, "yes")],
        [_description("desc-1", 42, [sentence])],
        [_polygon(42)],
    )

    assert result.overlap_rows == [
        {
            "source_pbf": SOURCE_PBF,
            "osm_type": "way",
            "osm_id": 42,
            "description_identity": "desc-1",
            "tag_key": "description",
            "sentence_index": 0,
            "sentence": sentence,
            "text_sha256": _hash(sentence),
            "language_code": "eng",
            "eunis_code": "T11",
            "eunis_name": "Temperate riparian forest",
            "eunis_overlap_percentage": 72.5,
            "eunis_source_version": "EEA EUNIS habitat probability maps v1 2021",
        }
    ]


def test_counts_negative_missing_and_unsplit_rows_without_inventing_sentences() -> None:
    unsplit_tag = "description:en"
    unsplit_label = _label(
        "desc-4", 0, "whole unsplit value", "skipped_unsplit", tag_key=unsplit_tag
    )
    unsplit_label["language"] = "tgl"
    result = _process(
        [
            _label("desc-1", 0, "  meadow\nedge ", "yes"),
            _label("desc-1", 1, "Not habitat evidence.", "no", language="eng"),
            _label("desc-2", 0, "A bog sentence.", "yes"),
            _label("desc-3", 0, "Unparsed model answer.", "failed"),
            unsplit_label,
        ],
        [
            _description("desc-1", 42, ["  meadow\nedge ", "Not habitat evidence."]),
            _description("desc-2", 43, ["A bog sentence."]),
            _description("desc-3", 44, ["Unparsed model answer."]),
            _description("desc-4", 45, [], tag_key=unsplit_tag, language="tgl"),
        ],
        [_polygon(42), _polygon(43, code=None), _polygon(44), _polygon(45, code=None)],
    )

    assert len(result.overlap_rows) == 1
    assert result.overlap_rows[0]["sentence"] == "  meadow\nedge "
    assert result.audit.decisions == {"yes": 2, "no": 1, "failed": 1, "skipped_unsplit": 1}
    assert result.audit.sentence_rows == 4
    assert result.audit.skipped_unsplit_rows == 1
    assert result.audit.yes_without_eunis_rows == 1
    assert result.audit.decision_assignment[("yes", "assigned")] == 1
    assert result.audit.decision_assignment[("yes", "missing")] == 1
    assert result.audit.decision_assignment[("skipped_unsplit", "missing")] == 1
    assert result.audit.unique_sentence_hashes == {
        _hash("  meadow\nedge "),
        _hash("Not habitat evidence."),
        _hash("A bog sentence."),
        _hash("Unparsed model answer."),
    }
    assert result.audit.overlap_by_tag == {"description": 1}
    assert result.audit.overlap_by_language == {"eng": 1}


def test_identical_sentence_text_keeps_distinct_occurrences() -> None:
    sentence = "same text in two separate places"
    result = _process(
        [_label("desc-1", 0, sentence, "yes"), _label("desc-2", 0, sentence, "yes")],
        [_description("desc-1", 42, [sentence]), _description("desc-2", 43, [sentence])],
        [_polygon(42), _polygon(43)],
    )

    assert len(result.overlap_rows) == 2
    assert result.audit.unique_sentence_hashes == {_hash(sentence)}
    assert {row["osm_id"] for row in result.overlap_rows} == {42, 43}


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda row: row.update(description_identity=None), DataValidationError),
        (lambda row: row.update(tag_key=None), DataValidationError),
        (lambda row: row.update(sentence_index=None), DataValidationError),
        (lambda row: row.update(sentence_index=True), DataValidationError),
        (lambda row: row.update(decision="maybe"), DataValidationError),
        (lambda row: row.update(input_revision="another-revision"), ProvenanceError),
        (lambda row: row.update(text_sha256="0" * 64), DataValidationError),
        (lambda row: row.update(text_sha256="z" * 64), DataValidationError),
        (lambda row: row.update(text_sha256="short"), DataValidationError),
        (lambda row: row.update(language="fra"), ProvenanceError),
        (lambda row: row.update(language=7), DataValidationError),
    ],
)
def test_invalid_label_values_fail_closed(mutate: Any, error: type[Exception]) -> None:
    sentence = "A grassland sentence."
    label = _label("desc-1", 0, sentence, "yes")
    mutate(label)

    with pytest.raises(error):
        _process([label], [_description("desc-1", 42, [sentence])], [_polygon(42)])


def test_unmatched_label_description_and_polygon_keys_are_rejected() -> None:
    sentence = "A grassland sentence."
    with pytest.raises(CardinalityError, match="unmatched description key"):
        _process([_label("missing", 0, sentence, "yes")], [], [_polygon(42)])
    with pytest.raises(CardinalityError, match="unmatched EUNIS polygon"):
        _process(
            [_label("desc-1", 0, sentence, "yes")], [_description("desc-1", 42, [sentence])], []
        )


def test_missing_sentence_label_is_not_silently_ignored() -> None:
    with pytest.raises(CardinalityError, match="1 labels for 2 sentence positions"):
        _process(
            [_label("desc-1", 0, "first", "yes")],
            [_description("desc-1", 42, ["first", "second"])],
            [_polygon(42)],
        )


def test_out_of_range_sentence_index_fails() -> None:
    with pytest.raises(DataValidationError, match="outside the sentence list"):
        _process(
            [_label("desc-1", 1, "only sentence", "yes")],
            [_description("desc-1", 42, ["only sentence"])],
            [_polygon(42)],
        )


def test_unsplit_status_must_use_empty_zero_index_sentinel() -> None:
    with pytest.raises(DataValidationError, match="skipped_unsplit"):
        _process(
            [_label("desc-1", 0, "a sentence", "skipped_unsplit")],
            [_description("desc-1", 42, ["a sentence"])],
            [_polygon(42)],
        )


@pytest.mark.parametrize(
    "percentage",
    [float("nan"), float("inf"), -0.01, 100.01, None, "72.5"],
)
def test_invalid_eunis_percentage_fails(percentage: Any) -> None:
    polygon = _polygon(42)
    polygon["eunis_overlap_percentage"] = percentage

    with pytest.raises(DataValidationError, match=r"overlap percentage|numeric overlap"):
        _process(
            [_label("desc-1", 0, "A grassland sentence.", "yes")],
            [_description("desc-1", 42, ["A grassland sentence."])],
            [polygon],
        )


def test_inconsistent_missing_eunis_fields_fail() -> None:
    polygon = _polygon(42, code=None)
    polygon["eunis_name"] = "stray name"
    with pytest.raises(DataValidationError, match="non-null companion"):
        _process(
            [_label("desc-1", 0, "A grassland sentence.", "yes")],
            [_description("desc-1", 42, ["A grassland sentence."])],
            [polygon],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eunis_code", " "),
        ("eunis_name", ""),
        ("eunis_overlap_percentage", True),
        ("eunis_source_version", ""),
    ],
)
def test_invalid_assigned_eunis_fields_fail(field: str, value: Any) -> None:
    polygon = _polygon(42)
    polygon[field] = value

    with pytest.raises(DataValidationError):
        _process(
            [_label("desc-1", 0, "A grassland sentence.", "yes")],
            [_description("desc-1", 42, ["A grassland sentence."])],
            [polygon],
        )


def test_assigned_eunis_code_must_be_text() -> None:
    polygon = _polygon(42)
    polygon["eunis_code"] = 7

    with pytest.raises(
        DataValidationError,
        match=r"tuvalu-latest.parquet polygon: eunis_code",
    ):
        _process(
            [_label("desc-1", 0, "A grassland sentence.", "yes")],
            [_description("desc-1", 42, ["A grassland sentence."])],
            [polygon],
        )


def test_invalid_polygon_ids_and_sentence_lists_fail() -> None:
    with pytest.raises(DataValidationError, match="positive"):
        _process([], [], [_polygon(0)])
    bad_id = _polygon(42)
    bad_id["osm_id"] = True
    with pytest.raises(DataValidationError, match="must be an integer"):
        _process([], [], [bad_id])

    sentence = "A grassland sentence."
    cases = [("sentences", "not a list", 1), ("sentences", [None], 1), ("sentences", [sentence], 2)]
    cases.extend(
        [("sentences", 17, 1), ("sentences", [sentence], "1"), ("sentences", [sentence], True)]
    )
    for field, value, count in cases:
        description = _description("desc-1", 42, [sentence])
        description[field] = value
        description["sentence_count"] = count
        with pytest.raises(DataValidationError):
            _process([], [description], [_polygon(42)])


def test_invalid_description_language_and_global_polygon_match_fail() -> None:
    sentence = "A grassland sentence."
    description = _description("desc-1", 42, [sentence])
    description["language_code"] = 17
    with pytest.raises(DataValidationError, match="language_code"):
        _process([_label("desc-1", 0, sentence, "yes")], [description], [_polygon(42)])


def test_null_description_and_label_languages_are_allowed() -> None:
    sentence = "A grassland sentence."
    description = _description("desc-1", 42, [sentence], language=None)
    label = _label("desc-1", 0, sentence, "yes", language="eng")

    result = _process([label], [description], [_polygon(42)])

    assert result.overlap_rows[0]["language_code"] is None

    description["language_code"] = "eng"
    label["language"] = None
    result = _process([label], [description], [_polygon(42)])
    assert result.overlap_rows[0]["language_code"] == "eng"


def test_unmatched_polygon_lookup_fails_closed() -> None:
    description = _description("desc-1", 42, ["a sentence"])

    with pytest.raises(CardinalityError, match="unmatched"):
        _polygon_for_description(description, {}, "test label")


def test_source_pbf_mismatch_is_a_provenance_failure() -> None:
    description = _description("desc-1", 42, ["A grassland sentence."])
    description["source_pbf"] = "wrong-region.osm.pbf"
    with pytest.raises(ProvenanceError, match="source_pbf"):
        _process(
            [_label("desc-1", 0, "A grassland sentence.", "yes")],
            [description],
            [_polygon(42)],
        )


def test_duplicate_polygon_description_and_sentence_keys_fail() -> None:
    sentence = "A grassland sentence."
    label = _label("desc-1", 0, sentence, "yes")
    description = _description("desc-1", 42, [sentence])
    polygon = _polygon(42)
    with pytest.raises(CardinalityError, match="duplicate global polygon key"):
        _process([label], [description], [polygon, polygon])
    with pytest.raises(CardinalityError, match="duplicate global description key"):
        _process([label], [description, description], [polygon])
    with pytest.raises(CardinalityError, match="duplicate sentence occurrence key"):
        _process([label, label], [description], [polygon])


def test_global_polygon_uniqueness_carries_across_partitions() -> None:
    sentence = "A grassland sentence."
    registry = GlobalKeys()
    _process(
        [_label("desc-1", 0, sentence, "yes")],
        [_description("desc-1", 42, [sentence])],
        [_polygon(42)],
        global_keys=registry,
    )
    second_description = _description("desc-2", 42, [sentence])
    second_description["source_pbf"] = "another-region.osm.pbf"
    second_polygon = _polygon(42)
    second_polygon["source_pbf"] = "another-region.osm.pbf"

    with pytest.raises(CardinalityError, match="duplicate global polygon key"):
        _process(
            [_label("desc-2", 0, sentence, "yes")],
            [second_description],
            [second_polygon],
            global_keys=registry,
        )
