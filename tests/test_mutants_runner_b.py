"""Exact-outcome tests for the pilot runner's freeze and read helpers."""

from __future__ import annotations

import errno
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset_text_label_benchmark.output_schema import LABELLED_SCHEMA
from georeset_text_label_benchmark.pilot.geo_sampling import h3_cell_of, h3_centre_of
from georeset_text_label_benchmark.pilot.runner import (
    _is_sha256,
    _read_frozen_sample,
    _read_pipeline_manifest,
    _read_source_rows,
    _require_balanced_decisions,
    _require_cell_centres,
    _sha256_json,
    _validate_candidate_count,
    _validate_freeze_inputs,
    _validate_frozen_source,
    _validate_sampled_candidate_names,
    _write_json_exclusive,
    sha256_file,
)

LABELLED = "labelled-eunis.parquet"


def _source_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_pbf": "test-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": 1,
        "description_identity": "description-1",
        "tag_key": "description",
        "sentence_index": 0,
        "sentence": "A snow-bed sentence.",
        "text_sha256": "a" * 64,
        "language_code": "eng",
        "eunis_code": "R41",
        "eunis_name": "Snow-bed vegetation",
        "eunis_overlap_percentage": 80.0,
        "eunis_source_version": "maps-v1",
        "decision": "yes",
        "bbox_min_x": 0.0,
        "bbox_min_y": 0.0,
        "bbox_max_x": 1.0,
        "bbox_max_y": 1.0,
    }
    row.update(overrides)
    return row


def test_sha256_file_returns_the_lowercase_digest_of_the_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_file_streams_large_files_to_the_same_digest(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 40_000
    path = tmp_path / "large.bin"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_of_an_empty_file_is_the_empty_digest(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert sha256_file(path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_file_raises_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.bin")


def test_sha256_json_hashes_sorted_compact_unescaped_utf8() -> None:
    expected = hashlib.sha256('{"a":"é","b":[1,2]}'.encode()).hexdigest()

    assert _sha256_json({"b": [1, 2], "a": "é"}) == expected


def test_sha256_json_uses_compact_separators_for_lists() -> None:
    assert _sha256_json(["x", "y"]) == hashlib.sha256(b'["x","y"]').hexdigest()


def test_write_json_exclusive_writes_sorted_two_space_json_with_one_trailing_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.json"

    _write_json_exclusive(path, {"b": 1, "a": "é"})

    assert path.read_bytes() == b'{\n  "a": "\xc3\xa9",\n  "b": 1\n}\n'


def test_write_json_exclusive_indents_nested_objects_by_two_spaces(tmp_path: Path) -> None:
    path = tmp_path / "nested.json"

    _write_json_exclusive(path, {"a": {"z": 1}})

    assert path.read_bytes() == b'{\n  "a": {\n    "z": 1\n  }\n}\n'


def test_write_json_exclusive_never_overwrites_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError) as error:
        _write_json_exclusive(path, {"a": 1})

    assert error.value.errno == errno.EEXIST
    assert path.read_text(encoding="utf-8") == "keep me"


def test_read_source_rows_returns_only_the_source_columns_and_the_exact_row_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labelled.parquet"
    schema = LABELLED_SCHEMA.append(pa.field("extra_column", pa.string()))
    rows = [
        {**_source_row(sentence="first sentence"), "extra_column": "x"},
        {**_source_row(sentence="second sentence", osm_id=2, decision="no"), "extra_column": "y"},
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)

    read_rows, count = _read_source_rows(path)

    assert count == 2
    assert len(read_rows) == 2
    assert all(set(row) == set(LABELLED_SCHEMA.names) for row in read_rows)
    assert [row["sentence"] for row in read_rows] == ["first sentence", "second sentence"]
    assert [row["decision"] for row in read_rows] == ["yes", "no"]


def test_read_pipeline_manifest_returns_the_parsed_mapping(tmp_path: Path) -> None:
    path = tmp_path / "pipeline-manifest.json"
    payload = {
        "artifact_sha256": {LABELLED: "a" * 64},
        "computation_commit": "c" * 40,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert dict(_read_pipeline_manifest(path)) == payload


def test_read_pipeline_manifest_reports_a_missing_file_with_the_exact_message(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"^pipeline manifest\.json is unreadable$") as error:
        _read_pipeline_manifest(tmp_path / "missing.json")

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_read_pipeline_manifest_reports_a_directory_path_as_unreadable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^pipeline manifest\.json is unreadable$") as error:
        _read_pipeline_manifest(tmp_path)

    assert isinstance(error.value.__cause__, OSError)


def test_read_pipeline_manifest_reports_invalid_json_as_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "pipeline-manifest.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"^pipeline manifest\.json is unreadable$") as error:
        _read_pipeline_manifest(path)

    assert isinstance(error.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '"text"',
        "{}",
        '{"artifact_sha256": []}',
        '{"artifact_sha256": "abc"}',
    ],
)
def test_read_pipeline_manifest_requires_a_mapping_of_artifact_checksums(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "pipeline-manifest.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"^pipeline manifest\.json must record artifact checksums$"
    ):
        _read_pipeline_manifest(path)


@pytest.mark.parametrize("count", [0, 157, 159])
def test_validate_candidate_count_rejects_any_count_other_than_158(count: int) -> None:
    candidates = [{"eunis_code": str(index)} for index in range(count)]

    with pytest.raises(ValueError, match=r"^expected 158 candidate classes$"):
        _validate_candidate_count(candidates)


def test_validate_candidate_count_accepts_exactly_158_candidates() -> None:
    _validate_candidate_count([{"eunis_code": str(index)} for index in range(158)])


def test_is_sha256_accepts_exactly_64_lowercase_hex_digits_in_a_string() -> None:
    assert _is_sha256("0123456789abcdef" * 4) is True
    assert _is_sha256("a" * 64) is True


@pytest.mark.parametrize(
    "value",
    ["a" * 63, "a" * 65, "A" * 64, "g" * 64, "X" * 64, "", None, 123, ["a" * 64]],
)
def test_is_sha256_rejects_every_other_value(value: Any) -> None:
    assert _is_sha256(value) is False


def test_validate_frozen_source_accepts_the_labelled_pool_with_both_checksums() -> None:
    _validate_frozen_source(
        {"file": LABELLED, "sha256": "a" * 64, "pipeline_manifest_sha256": "b" * 64}
    )


def test_validate_frozen_source_rejects_a_sample_from_another_file() -> None:
    with pytest.raises(ValueError, match=r"^frozen sample is not from the labelled EUNIS pool$"):
        _validate_frozen_source(
            {"file": "overlap.parquet", "sha256": "a" * 64, "pipeline_manifest_sha256": "b" * 64}
        )


@pytest.mark.parametrize(
    "source",
    [
        {"file": LABELLED, "sha256": "a" * 64},
        {"file": LABELLED, "pipeline_manifest_sha256": "b" * 64},
        {"file": LABELLED, "sha256": "a" * 63, "pipeline_manifest_sha256": "b" * 64},
        {"file": LABELLED, "sha256": "a" * 64, "pipeline_manifest_sha256": "B" * 64},
    ],
)
def test_validate_frozen_source_requires_both_sha256_checksums(source: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"^frozen sample source checksums are missing$"):
        _validate_frozen_source(source)


def test_validate_sampled_candidate_names_accepts_matching_names() -> None:
    sample = {
        "selected_rows": [
            {"eunis_code": "R41", "eunis_name": "Snow-bed vegetation"},
            {"eunis_code": "R42", "eunis_name": "Other vegetation"},
        ]
    }

    _validate_sampled_candidate_names(
        sample, {"R41": "Snow-bed vegetation", "R42": "Other vegetation"}
    )


def test_validate_sampled_candidate_names_reports_the_first_mismatched_code() -> None:
    sample = {
        "selected_rows": [
            {"eunis_code": "R41", "eunis_name": "Snow-bed vegetation"},
            {"eunis_code": "R42", "eunis_name": "Renamed"},
        ]
    }

    with pytest.raises(ValueError, match=r"^frozen gold name mismatch for R42$"):
        _validate_sampled_candidate_names(
            sample, {"R41": "Snow-bed vegetation", "R42": "Other vegetation"}
        )


def test_validate_sampled_candidate_names_reports_a_code_missing_from_the_vocabulary() -> None:
    sample = {"selected_rows": [{"eunis_code": "R41", "eunis_name": "Snow-bed vegetation"}]}

    with pytest.raises(ValueError, match=r"^frozen gold name mismatch for R41$"):
        _validate_sampled_candidate_names(sample, {})


def _freeze_input_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    labelled = tmp_path / LABELLED
    labelled.write_bytes(b"labelled bytes")
    candidate = tmp_path / "candidate_labels.csv"
    candidate.write_bytes(b"candidate bytes")
    manifest = tmp_path / "pipeline-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_sha256": {
                    LABELLED: hashlib.sha256(b"labelled bytes").hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    return labelled, candidate, manifest


def test_validate_freeze_inputs_returns_both_hashes_and_the_manifest(tmp_path: Path) -> None:
    labelled, candidate, manifest_path = _freeze_input_files(tmp_path)

    labelled_hash, candidate_hash, manifest = _validate_freeze_inputs(
        labelled, candidate, manifest_path, None
    )

    assert labelled_hash == hashlib.sha256(b"labelled bytes").hexdigest()
    assert candidate_hash == hashlib.sha256(b"candidate bytes").hexdigest()
    assert dict(manifest) == {
        "artifact_sha256": {LABELLED: hashlib.sha256(b"labelled bytes").hexdigest()}
    }


def test_validate_freeze_inputs_accepts_a_candidate_table_with_the_pinned_hash(
    tmp_path: Path,
) -> None:
    labelled, candidate, manifest_path = _freeze_input_files(tmp_path)
    pinned = hashlib.sha256(b"candidate bytes").hexdigest()

    _labelled_hash, candidate_hash, _manifest = _validate_freeze_inputs(
        labelled, candidate, manifest_path, pinned
    )

    assert candidate_hash == pinned


def test_validate_freeze_inputs_rejects_a_labelled_pool_the_manifest_does_not_name(
    tmp_path: Path,
) -> None:
    labelled, candidate, manifest_path = _freeze_input_files(tmp_path)
    labelled.write_bytes(b"changed bytes")

    with pytest.raises(
        ValueError, match=r"^labelled-eunis\.parquet SHA-256 does not match the pipeline manifest$"
    ):
        _validate_freeze_inputs(labelled, candidate, manifest_path, None)


def test_validate_freeze_inputs_rejects_a_manifest_without_the_labelled_checksum(
    tmp_path: Path,
) -> None:
    labelled, candidate, manifest_path = _freeze_input_files(tmp_path)
    manifest_path.write_text(json.dumps({"artifact_sha256": {}}), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"^labelled-eunis\.parquet SHA-256 does not match the pipeline manifest$"
    ):
        _validate_freeze_inputs(labelled, candidate, manifest_path, None)


def test_validate_freeze_inputs_rejects_a_candidate_table_off_the_pinned_hash(
    tmp_path: Path,
) -> None:
    labelled, candidate, manifest_path = _freeze_input_files(tmp_path)

    with pytest.raises(
        ValueError, match=r"^candidate CSV SHA-256 does not match the pinned input$"
    ):
        _validate_freeze_inputs(labelled, candidate, manifest_path, "0" * 64)


def _balanced_rows(yes: int, no: int) -> list[dict[str, str]]:
    return [{"decision": "yes"}] * yes + [{"decision": "no"}] * no


def test_require_balanced_decisions_accepts_exactly_per_group_of_each() -> None:
    _require_balanced_decisions(_balanced_rows(3, 3), 3)


@pytest.mark.parametrize(("yes", "no"), [(3, 2), (2, 3), (3, 1), (1, 3), (0, 6), (6, 0)])
def test_require_balanced_decisions_rejects_any_other_split(yes: int, no: int) -> None:
    with pytest.raises(ValueError, match=r"^frozen sample must have 3 yes and 3 no$"):
        _require_balanced_decisions(_balanced_rows(yes, no), 3)


def _centred_row(cell: str, lat_shift: float = 0.0) -> dict[str, Any]:
    lat, lon = h3_centre_of(cell)
    return {"h3_cell": cell, "cell_centre_lat": lat + lat_shift, "cell_centre_lon": lon}


def test_require_cell_centres_accepts_rows_whose_centre_is_their_cell_centre() -> None:
    cell = h3_cell_of(0.0, 0.0)

    _require_cell_centres([_centred_row(cell), _centred_row(cell)])


def test_require_cell_centres_rejects_a_centre_that_is_not_its_cell_centre() -> None:
    cell = h3_cell_of(0.0, 0.0)

    with pytest.raises(ValueError, match=r"^frozen sample H3 cell does not match its centre$"):
        _require_cell_centres([_centred_row(cell, lat_shift=1.0)])


def test_require_cell_centres_checks_every_row_not_only_the_first() -> None:
    cell = h3_cell_of(0.0, 0.0)

    with pytest.raises(ValueError, match=r"^frozen sample H3 cell does not match its centre$"):
        _require_cell_centres([_centred_row(cell), _centred_row(cell, lat_shift=1.0)])


def _ascii_default_text_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an unspecified text encoding resolve to ASCII, as on an ASCII-locale host."""
    real_text_encoding = io.text_encoding

    def text_encoding(encoding: str | None, stacklevel: int = 2) -> str:
        return "ascii" if encoding is None else real_text_encoding(encoding, stacklevel)

    monkeypatch.setattr(io, "text_encoding", text_encoding)


def test_read_pipeline_manifest_decodes_utf8_when_the_default_encoding_is_ascii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"artifact_sha256": {LABELLED: "a" * 64}, "note": "café"}
    path = tmp_path / "pipeline-manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _ascii_default_text_encoding(monkeypatch)

    assert dict(_read_pipeline_manifest(path)) == payload


def _frozen_row(sample_id: str, decision: str, cell: str, osm_id: int) -> dict[str, Any]:
    lat, lon = h3_centre_of(cell)
    return {
        "sample_id": sample_id,
        "text_sha256": hashlib.sha256(sample_id.encode("utf-8")).hexdigest(),
        "source_pbf": "test-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": osm_id,
        "decision": decision,
        "h3_cell": cell,
        "cell_centre_lat": lat,
        "cell_centre_lon": lon,
    }


def _frozen_sample_document(note: str) -> dict[str, Any]:
    rows = [
        _frozen_row("sample-yes", "yes", h3_cell_of(0.0, 0.0), 1),
        _frozen_row("sample-no", "no", h3_cell_of(40.0, 40.0), 2),
    ]
    return {
        "note": note,
        "source": {"file": LABELLED, "sha256": "a" * 64, "pipeline_manifest_sha256": "b" * 64},
        "selection": {
            "sample_size": 2,
            "per_group": 1,
            "sample_ids_sha256": _sha256_json([row["sample_id"] for row in rows]),
        },
        "selected_rows": rows,
    }


def _write_frozen_sample(run_dir: Path, document: dict[str, Any]) -> None:
    run_dir.mkdir()
    (run_dir / "frozen_sample.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )


def test_read_frozen_sample_returns_the_document_and_its_selected_rows(tmp_path: Path) -> None:
    _write_frozen_sample(tmp_path / "run", _frozen_sample_document("pilot"))

    sample, rows = _read_frozen_sample(tmp_path / "run")

    assert sample["note"] == "pilot"
    assert [row["sample_id"] for row in rows] == ["sample-yes", "sample-no"]
    assert [row["decision"] for row in rows] == ["yes", "no"]


def test_read_frozen_sample_decodes_utf8_when_the_default_encoding_is_ascii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_frozen_sample(tmp_path / "run", _frozen_sample_document("café"))
    _ascii_default_text_encoding(monkeypatch)

    sample, rows = _read_frozen_sample(tmp_path / "run")

    assert sample["note"] == "café"
    assert [row["sample_id"] for row in rows] == ["sample-yes", "sample-no"]
