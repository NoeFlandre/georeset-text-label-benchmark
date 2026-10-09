"""Contract tests for occurrence-preserving Description joins."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pytest

from georeset_text_label_benchmark.errors import (
    CardinalityError,
    DataValidationError,
    ProvenanceError,
)
from georeset_text_label_benchmark.join import (
    _join_labels,
    _polygon_for_description,
    _sentence_for_label,
    process_partition,
)
from georeset_text_label_benchmark.models import AuditCounts, GlobalKeys

INPUT_REVISION = "b4706eb315c66a8a57135289e7612dca8e03caf8"
SOURCE_PBF = "tuvalu-latest.osm.pbf"
EUNIS_REFERENCE_VERSION = "EEA EUNIS habitat probability maps v1 2021"


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
        "eunis_source_version": EUNIS_REFERENCE_VERSION if code else None,
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
        expected_eunis_source_version=EUNIS_REFERENCE_VERSION,
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


def test_repeated_decisions_increment_all_partition_audit_counters() -> None:
    labels = []
    descriptions = []
    polygons = []
    for osm_id in (42, 43):
        sentence = f"Assigned sentence {osm_id}."
        labels.append(_label(f"assigned-{osm_id}", 0, sentence, "yes"))
        descriptions.append(_description(f"assigned-{osm_id}", osm_id, [sentence]))
        polygons.append(_polygon(osm_id))
    for osm_id in (44, 45):
        sentence = f"Unassigned sentence {osm_id}."
        labels.append(_label(f"missing-{osm_id}", 0, sentence, "yes"))
        descriptions.append(_description(f"missing-{osm_id}", osm_id, [sentence]))
        polygons.append(_polygon(osm_id, code=None))
    for osm_id in (46, 47):
        tag_key = "description:en"
        labels.append(
            _label(
                f"unsplit-{osm_id}", 0, "whole unsplit value", "skipped_unsplit", tag_key=tag_key
            )
        )
        descriptions.append(_description(f"unsplit-{osm_id}", osm_id, [], tag_key=tag_key))
        polygons.append(_polygon(osm_id, code=None))

    result = _process(labels, descriptions, polygons)

    assert result.audit.decision_assignment == {
        ("yes", "assigned"): 2,
        ("yes", "missing"): 2,
        ("skipped_unsplit", "missing"): 2,
    }
    assert result.audit.skipped_unsplit_rows == 2
    assert result.audit.yes_without_eunis_rows == 2
    assert result.audit.retained_rows == 2
    assert result.audit.overlap_by_eunis == {"T11": 2}
    assert result.audit.overlap_by_tag == {"description": 2}
    assert result.audit.overlap_by_language == {"eng": 2}


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
    assert result.audit.decision_assignment[("yes", "assigned")] == 2


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (
            lambda row: row.update(description_identity=None),
            DataValidationError,
            "tuvalu-latest.parquet label: description_identity must be a non-empty string",
        ),
        (
            lambda row: row.update(description_identity=7),
            DataValidationError,
            "tuvalu-latest.parquet label: description_identity must be a non-empty string",
        ),
        (
            lambda row: row.update(tag_key=None),
            DataValidationError,
            "tuvalu-latest.parquet label: tag_key must be a non-empty string",
        ),
        (
            lambda row: row.update(tag_key=7),
            DataValidationError,
            "tuvalu-latest.parquet label: tag_key must be a non-empty string",
        ),
        (
            lambda row: row.update(sentence_index=None),
            DataValidationError,
            "tuvalu-latest.parquet label: sentence_index must be an integer",
        ),
        (
            lambda row: row.update(sentence_index=True),
            DataValidationError,
            "tuvalu-latest.parquet label: sentence_index must be an integer",
        ),
        (
            lambda row: row.update(decision="maybe"),
            DataValidationError,
            "tuvalu-latest.parquet label: unrecognized decision 'maybe'",
        ),
        (
            lambda row: row.update(decision=None),
            DataValidationError,
            "tuvalu-latest.parquet label: decision must be a non-empty string",
        ),
        (
            lambda row: row.update(input_revision="another-revision"),
            ProvenanceError,
            "tuvalu-latest.parquet label: input_revision does not match pinned Description source",
        ),
        (
            lambda row: row.update(input_revision=None),
            DataValidationError,
            "tuvalu-latest.parquet label: input_revision must be a non-empty string",
        ),
        (
            lambda row: row.update(text_sha256="0" * 64),
            DataValidationError,
            "tuvalu-latest.parquet label: text_sha256 does not match exact sentence bytes",
        ),
        (
            lambda row: row.update(text_sha256=None),
            DataValidationError,
            "tuvalu-latest.parquet label: text_sha256 must be a non-empty string",
        ),
        (
            lambda row: row.update(text_sha256="z" * 64),
            DataValidationError,
            "tuvalu-latest.parquet label: text_sha256 must contain 64 hexadecimal characters",
        ),
        (
            lambda row: row.update(text_sha256="short"),
            DataValidationError,
            "tuvalu-latest.parquet label: text_sha256 must be a 64-character SHA-256 digest",
        ),
        (
            lambda row: row.update(language="fra"),
            ProvenanceError,
            "tuvalu-latest.parquet label: label language differs from detected language_code",
        ),
        (
            lambda row: row.update(language=7),
            DataValidationError,
            "tuvalu-latest.parquet label: label language must be a string or null",
        ),
    ],
)
def test_invalid_label_values_fail_closed(
    mutate: Any, error: type[Exception], message: str
) -> None:
    sentence = "A grassland sentence."
    label = _label("desc-1", 0, sentence, "yes")
    mutate(label)

    with pytest.raises(error, match=re.escape(message)):
        _process([label], [_description("desc-1", 42, [sentence])], [_polygon(42)])


def test_unmatched_label_description_and_polygon_keys_are_rejected() -> None:
    sentence = "A grassland sentence."
    with pytest.raises(CardinalityError, match="unmatched description key"):
        _process([_label("missing", 0, sentence, "yes")], [], [_polygon(42)])
    with pytest.raises(CardinalityError, match="unmatched EUNIS polygon"):
        _process(
            [_label("desc-1", 0, sentence, "yes")], [_description("desc-1", 42, [sentence])], []
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "description_identity",
            None,
            "tuvalu-latest.parquet description-tag: description_identity must be a non-empty string",
        ),
        (
            "tag_key",
            None,
            "tuvalu-latest.parquet description-tag: tag_key must be a non-empty string",
        ),
        (
            "osm_id",
            True,
            "tuvalu-latest.parquet description-tag: osm_id must be an integer",
        ),
    ],
)
def test_description_key_and_polygon_validation_keep_context(
    field: str, value: Any, message: str
) -> None:
    sentence = "A grassland sentence."
    description = _description("desc-1", 42, [sentence])
    description[field] = value

    with pytest.raises(DataValidationError, match=re.escape(message)) as caught:
        _process([_label("desc-1", 0, sentence, "yes")], [description], [_polygon(42)])

    assert str(caught.value) == message


def test_description_polygon_lookup_keeps_context() -> None:
    sentence = "A grassland sentence."

    with pytest.raises(
        CardinalityError,
        match=re.escape(
            "tuvalu-latest.parquet description-tag: unmatched EUNIS polygon key ('way', 42)"
        ),
    ) as caught:
        _process(
            [_label("desc-1", 0, sentence, "yes")],
            [_description("desc-1", 42, [sentence])],
            [],
        )

    assert (
        str(caught.value)
        == "tuvalu-latest.parquet description-tag: unmatched EUNIS polygon key ('way', 42)"
    )


def test_sentence_lookup_revalidation_keeps_context() -> None:
    description = _description("desc-1", 42, ["A grassland sentence."])
    description["sentence_count"] = 2

    with pytest.raises(
        DataValidationError,
        match=re.escape(
            "tuvalu-latest.parquet label: sentence_count does not match sentences length"
        ),
    ) as caught:
        _sentence_for_label(description, 0, "yes", "tuvalu-latest.parquet label")

    assert str(caught.value) == (
        "tuvalu-latest.parquet label: sentence_count does not match sentences length"
    )


def test_label_phase_polygon_lookup_keeps_context() -> None:
    sentence = "A grassland sentence."
    description = _description("desc-1", 42, [sentence])
    labels = [_label("desc-1", 0, sentence, "yes")]

    with pytest.raises(
        CardinalityError,
        match=re.escape(
            "tuvalu-latest.parquet label: description polygon key ('way', 42) is unmatched"
        ),
    ) as caught:
        _join_labels(
            labels,
            {("desc-1", "description"): description},
            {},
            GlobalKeys(),
            AuditCounts(),
            INPUT_REVISION,
            "tuvalu-latest.parquet",
            labelled=False,
        )

    assert str(caught.value) == (
        "tuvalu-latest.parquet label: description polygon key ('way', 42) is unmatched"
    )


def test_missing_sentence_label_is_not_silently_ignored() -> None:
    with pytest.raises(
        CardinalityError,
        match=r"tuvalu-latest.parquet: description \('desc-1', 'description'\) has 1 labels for 2 sentence positions",
    ):
        _process(
            [_label("desc-1", 0, "first", "yes")],
            [_description("desc-1", 42, ["first", "second"])],
            [_polygon(42)],
        )


def test_out_of_range_sentence_index_fails() -> None:
    with pytest.raises(
        DataValidationError,
        match=r"tuvalu-latest.parquet label: sentence_index 1 is outside the sentence list",
    ):
        _process(
            [_label("desc-1", 1, "only sentence", "yes")],
            [_description("desc-1", 42, ["only sentence"])],
            [_polygon(42)],
        )


def test_negative_sentence_index_fails() -> None:
    with pytest.raises(
        DataValidationError,
        match=r"tuvalu-latest.parquet label: sentence_index -1 is outside the sentence list",
    ):
        _process(
            [_label("desc-1", -1, "only sentence", "yes")],
            [_description("desc-1", 42, ["only sentence"])],
            [_polygon(42)],
        )


def test_unsplit_status_must_use_empty_zero_index_sentinel() -> None:
    with pytest.raises(
        DataValidationError,
        match=r"tuvalu-latest.parquet label: skipped_unsplit must be an empty index-0 sentinel",
    ):
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


@pytest.mark.parametrize("percentage", [0, 100])
def test_valid_eunis_overlap_percentage_includes_both_boundaries(percentage: int) -> None:
    polygon = _polygon(42)
    polygon["eunis_overlap_percentage"] = percentage

    result = _process(
        [_label("desc-1", 0, "A grassland sentence.", "yes")],
        [_description("desc-1", 42, ["A grassland sentence."])],
        [polygon],
    )

    assert result.overlap_rows[0]["eunis_overlap_percentage"] == percentage


def test_inconsistent_missing_eunis_fields_fail() -> None:
    for field, value in (
        ("eunis_name", "stray name"),
        ("eunis_overlap_percentage", 0),
        ("eunis_source_version", EUNIS_REFERENCE_VERSION),
    ):
        polygon = _polygon(42, code=None)
        polygon[field] = value
        with pytest.raises(
            DataValidationError,
            match=r"tuvalu-latest.parquet polygon: missing EUNIS code has non-null companion fields",
        ):
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
    messages = {
        "eunis_code": r"tuvalu-latest.parquet polygon: eunis_code must be a non-empty string",
        "eunis_name": r"tuvalu-latest.parquet polygon: eunis_name must be a non-empty string",
        "eunis_overlap_percentage": r"tuvalu-latest.parquet polygon: assigned EUNIS code needs a numeric overlap",
        "eunis_source_version": r"tuvalu-latest.parquet polygon: eunis_source_version must be a non-empty string",
    }
    with pytest.raises(DataValidationError, match=messages[field]):
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
    missing_pbf = _polygon(42)
    missing_pbf["source_pbf"] = None
    with pytest.raises(
        DataValidationError,
        match=re.escape("tuvalu-latest.parquet polygon: source_pbf must be a non-empty string"),
    ):
        _process([], [], [missing_pbf])

    bad_type = _polygon(42)
    bad_type["osm_type"] = None
    with pytest.raises(
        DataValidationError,
        match=r"tuvalu-latest.parquet polygon: osm_type must be a non-empty string",
    ):
        _process([], [], [bad_type])
    with pytest.raises(DataValidationError, match="positive"):
        _process([], [], [_polygon(0)])
    bad_id = _polygon(42)
    bad_id["osm_id"] = True
    with pytest.raises(
        DataValidationError,
        match=re.escape("tuvalu-latest.parquet polygon: osm_id must be an integer"),
    ):
        _process([], [], [bad_id])

    sentence = "A grassland sentence."
    cases = [
        (
            "sentences",
            "not a list",
            1,
            "tuvalu-latest.parquet description-tag: sentences must be a list of strings",
        ),
        (
            "sentences",
            [None],
            1,
            "tuvalu-latest.parquet description-tag: every sentence must be a string",
        ),
        (
            "sentences",
            [sentence],
            2,
            "tuvalu-latest.parquet description-tag: sentence_count does not match sentences length",
        ),
    ]
    cases.extend(
        [
            (
                "sentences",
                17,
                1,
                "tuvalu-latest.parquet description-tag: sentences must be a list of strings",
            ),
            (
                "sentences",
                [sentence],
                "1",
                "tuvalu-latest.parquet description-tag: sentence_count does not match sentences length",
            ),
            (
                "sentences",
                [sentence],
                True,
                "tuvalu-latest.parquet description-tag: sentence_count does not match sentences length",
            ),
        ]
    )
    for field, value, count, message in cases:
        description = _description("desc-1", 42, [sentence])
        description[field] = value
        description["sentence_count"] = count
        with pytest.raises(DataValidationError, match=re.escape(message)):
            _process([], [description], [_polygon(42)])


def test_invalid_description_language_and_global_polygon_match_fail() -> None:
    sentence = "A grassland sentence."
    description = _description("desc-1", 42, [sentence])
    description["language_code"] = 17
    with pytest.raises(
        DataValidationError,
        match=r"tuvalu-latest.parquet description-tag: language_code must be a string or null",
    ):
        _process([_label("desc-1", 0, sentence, "yes")], [description], [_polygon(42)])


def test_null_description_and_label_languages_are_allowed() -> None:
    sentence = "A grassland sentence."
    description = _description("desc-1", 42, [sentence], language=None)
    label = _label("desc-1", 0, sentence, "yes", language="eng")

    result = _process([label], [description], [_polygon(42)])

    assert result.overlap_rows[0]["language_code"] is None
    assert result.audit.overlap_by_language == {"unknown": 1}

    description["language_code"] = "eng"
    label["language"] = None
    result = _process([label], [description], [_polygon(42)])
    assert result.overlap_rows[0]["language_code"] == "eng"


def test_unmatched_polygon_lookup_fails_closed() -> None:
    description = _description("desc-1", 42, ["a sentence"])

    with pytest.raises(
        CardinalityError,
        match=re.escape("test label: description polygon key ('way', 42) is unmatched"),
    ) as caught:
        _polygon_for_description(description, {}, "test label")

    assert str(caught.value) == "test label: description polygon key ('way', 42) is unmatched"


def test_polygon_key_validation_keeps_label_context() -> None:
    description = _description("desc-1", 42, ["a sentence"])
    description["osm_id"] = None
    message = "test label: osm_id must be an integer"

    with pytest.raises(DataValidationError, match=re.escape(message)) as caught:
        _polygon_for_description(description, {}, "test label")

    assert str(caught.value) == message


def test_source_pbf_mismatch_is_a_provenance_failure() -> None:
    description = _description("desc-1", 42, ["A grassland sentence."])
    description["source_pbf"] = "wrong-region.osm.pbf"
    with pytest.raises(
        ProvenanceError,
        match=r"tuvalu-latest.parquet description-tag: description source_pbf 'wrong-region.osm.pbf' differs from polygon source_pbf 'tuvalu-latest.osm.pbf'",
    ):
        _process(
            [_label("desc-1", 0, "A grassland sentence.", "yes")],
            [description],
            [_polygon(42)],
        )


def test_description_source_pbf_validation_keeps_partition_context() -> None:
    sentence = "A grassland sentence."
    description = _description("desc-1", 42, [sentence])
    description["source_pbf"] = None
    message = "tuvalu-latest.parquet description-tag: source_pbf must be a non-empty string"

    with pytest.raises(DataValidationError, match=re.escape(message)) as caught:
        _process(
            [_label("desc-1", 0, sentence, "yes")],
            [description],
            [_polygon(42)],
        )

    assert str(caught.value) == message


def test_duplicate_polygon_description_and_sentence_keys_fail() -> None:
    sentence = "A grassland sentence."
    label = _label("desc-1", 0, sentence, "yes")
    description = _description("desc-1", 42, [sentence])
    polygon = _polygon(42)
    with pytest.raises(
        CardinalityError, match=r"tuvalu-latest.parquet polygon: duplicate global polygon key"
    ):
        _process([label], [description], [polygon, polygon])
    with pytest.raises(
        CardinalityError,
        match=r"tuvalu-latest.parquet description-tag: duplicate global description key",
    ):
        _process([label], [description, description], [polygon])
    with pytest.raises(
        CardinalityError,
        match=r"tuvalu-latest.parquet label: duplicate sentence occurrence key",
    ):
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


def test_global_description_key_uniqueness_carries_across_partitions() -> None:
    sentence = "A grassland sentence."
    registry = GlobalKeys()
    _process(
        [_label("desc-1", 0, sentence, "yes")],
        [_description("desc-1", 42, [sentence])],
        [_polygon(42)],
        global_keys=registry,
    )
    second = _description("desc-1", 43, [sentence])
    second["source_pbf"] = "another-region.osm.pbf"
    polygon = _polygon(43)
    polygon["source_pbf"] = "another-region.osm.pbf"

    with pytest.raises(
        CardinalityError,
        match=r"tuvalu-latest.parquet description-tag: duplicate global description key",
    ):
        _process(
            [_label("desc-1", 0, sentence, "yes")],
            [second],
            [polygon],
            global_keys=registry,
        )


def _bbox_polygon(osm_id: int, *, code: str | None = "T11") -> dict[str, Any]:
    return {
        **_polygon(osm_id, code=code),
        "bbox_min_x": 1.0,
        "bbox_min_y": 2.0,
        "bbox_max_x": 1.5,
        "bbox_max_y": 2.5,
    }


def _labelled(
    labels: list[dict[str, Any]],
    descriptions: list[dict[str, Any]],
    polygons: list[dict[str, Any]],
) -> Any:
    return process_partition(
        labels,
        descriptions,
        polygons,
        input_revision=INPUT_REVISION,
        partition_name="tuvalu-latest.parquet",
        expected_eunis_source_version=EUNIS_REFERENCE_VERSION,
        labelled=True,
    )


def test_default_mode_keeps_yes_rows_only_without_labelled_fields() -> None:
    result = _process(
        [_label("desc-1", 0, "Meadow.", "yes"), _label("desc-2", 0, "Town.", "no")],
        [_description("desc-1", 42, ["Meadow."]), _description("desc-2", 43, ["Town."])],
        [_polygon(42), _polygon(43)],
    )

    assert [row["osm_id"] for row in result.overlap_rows] == [42]
    assert "decision" not in result.overlap_rows[0]
    assert "bbox_min_x" not in result.overlap_rows[0]


def test_labelled_mode_keeps_assigned_yes_and_no_rows_with_decision_and_bbox() -> None:
    result = _labelled(
        [_label("desc-1", 0, "Meadow.", "yes"), _label("desc-2", 0, "Town.", "no")],
        [_description("desc-1", 42, ["Meadow."]), _description("desc-2", 43, ["Town."])],
        [_bbox_polygon(42), _bbox_polygon(43)],
    )

    assert [(row["decision"], row["osm_id"]) for row in result.overlap_rows] == [
        ("yes", 42),
        ("no", 43),
    ]
    no_row = result.overlap_rows[1]
    assert no_row["sentence"] == "Town."
    assert no_row["eunis_code"] == "T11"
    assert (no_row["bbox_min_x"], no_row["bbox_min_y"]) == (1.0, 2.0)
    assert (no_row["bbox_max_x"], no_row["bbox_max_y"]) == (1.5, 2.5)


def test_labelled_mode_drops_unassigned_rows_and_counts_only_yes_as_overlap() -> None:
    result = _labelled(
        [_label("desc-1", 0, "Meadow.", "yes"), _label("desc-2", 0, "Town.", "no")],
        [_description("desc-1", 42, ["Meadow."]), _description("desc-2", 43, ["Town."])],
        [_bbox_polygon(42, code=None), _bbox_polygon(43, code=None)],
    )

    assert result.overlap_rows == []
    assert result.audit.yes_without_eunis_rows == 1
    assert result.audit.retained_rows == 0


def test_labelled_mode_requires_polygon_bbox_fields() -> None:
    with pytest.raises(DataValidationError, match="bbox_min_x"):
        _labelled(
            [_label("desc-1", 0, "Meadow.", "yes")],
            [_description("desc-1", 42, ["Meadow."])],
            [_polygon(42)],
        )
