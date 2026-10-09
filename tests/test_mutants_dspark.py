"""Mutation-killing contracts for DSpark frozen-input checks and geographic sample IDs."""

from __future__ import annotations

import codecs
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from georeset_text_label_benchmark.pilot import dspark_runner
from georeset_text_label_benchmark.pilot.geo_sampling import (
    _haversine_km,
    _leaves_groups_feasible,
    _take,
    sample_id_of,
)

FROZEN_SAMPLE_BYTES = b'{"selected_rows": []}\n'
CANDIDATE_BYTES = b"eunis_code,eunis_name\nT11,Temperate forest\n"
SAMPLE_IDS_SHA256 = "a" * 64
MANIFEST_MISSING = "frozen manifest.json is required to verify the published inputs"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frozen_run(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Write a consistent frozen run directory and the sample it describes."""
    run_dir = tmp_path / "frozen"
    run_dir.mkdir()
    (run_dir / "frozen_sample.json").write_bytes(FROZEN_SAMPLE_BYTES)
    (run_dir / "candidate_labels.csv").write_bytes(CANDIDATE_BYTES)
    manifest = {
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "seed": 42,
        "sample_size": 100,
        "outputs_sha256": {
            "frozen_sample.json": _sha256_bytes(FROZEN_SAMPLE_BYTES),
            "candidate_labels.csv": _sha256_bytes(CANDIDATE_BYTES),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sample = {"selection": {"sample_ids_sha256": SAMPLE_IDS_SHA256, "seed": 42, "sample_size": 100}}
    return run_dir, sample


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _selection() -> dict[str, Any]:
    return {"sample_ids_sha256": SAMPLE_IDS_SHA256, "seed": 42, "sample_size": 100}


# _read_frozen_manifest


def test_frozen_manifest_read_returns_the_decoded_json_object(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"seed": 42, "label": "été"}', encoding="utf-8")

    manifest = dspark_runner._read_frozen_manifest(path)

    assert manifest == {"seed": 42, "label": "été"}
    assert isinstance(manifest, dict)


def test_missing_frozen_manifest_is_reported_with_the_exact_message(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=rf"^{re.escape(MANIFEST_MISSING)}$") as error:
        dspark_runner._read_frozen_manifest(tmp_path / "manifest.json")

    assert str(error.value) == MANIFEST_MISSING


def test_directory_at_the_manifest_path_is_reported_as_missing(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").mkdir()

    with pytest.raises(ValueError, match=rf"^{re.escape(MANIFEST_MISSING)}$") as error:
        dspark_runner._read_frozen_manifest(tmp_path / "manifest.json")

    assert str(error.value) == MANIFEST_MISSING


def test_invalid_frozen_manifest_json_is_unreadable_with_the_decode_cause(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"^frozen manifest\.json is unreadable$") as error:
        dspark_runner._read_frozen_manifest(path)

    assert str(error.value) == "frozen manifest.json is unreadable"
    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def test_os_failure_reading_frozen_manifest_keeps_the_os_error_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")

    def denied(self: Path, *args: Any, **kwargs: Any) -> str:
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "read_text", denied)

    with pytest.raises(ValueError, match=r"^frozen manifest\.json is unreadable$") as error:
        dspark_runner._read_frozen_manifest(path)

    assert str(error.value) == "frozen manifest.json is unreadable"
    assert isinstance(error.value.__cause__, PermissionError)


def test_frozen_manifest_is_decoded_as_utf8_whatever_the_locale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes('{"label": "été"}'.encode())
    original_read_text = Path.read_text
    encodings: list[Any] = []

    def recording_read_text(self: Path, *args: Any, encoding: Any = None, **kwargs: Any) -> str:
        encodings.append(encoding)
        return original_read_text(self, *args, encoding=encoding, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    assert dspark_runner._read_frozen_manifest(path) == {"label": "été"}
    assert [codecs.lookup(name).name for name in encodings] == ["utf-8"]


@pytest.mark.parametrize("text", ["[]", '"text"', "42", "null", "true"])
def test_frozen_manifest_must_decode_to_a_json_object(tmp_path: Path, text: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"^frozen manifest\.json must contain a JSON object$"
    ) as error:
        dspark_runner._read_frozen_manifest(path)

    assert str(error.value) == "frozen manifest.json must contain a JSON object"


# _verify_frozen_input_hashes


def test_frozen_input_hashes_pass_when_both_recorded_hashes_match(tmp_path: Path) -> None:
    run_dir, _ = _frozen_run(tmp_path)

    assert dspark_runner._verify_frozen_input_hashes(run_dir, _manifest(run_dir)) is None


@pytest.mark.parametrize("filename", ["frozen_sample.json", "candidate_labels.csv"])
def test_tampered_frozen_input_is_named_by_its_file(tmp_path: Path, filename: str) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    (run_dir / filename).write_bytes(b"tampered after freeze\n")

    with pytest.raises(
        ValueError, match=rf"^frozen manifest hash mismatch for {re.escape(filename)}$"
    ) as error:
        dspark_runner._verify_frozen_input_hashes(run_dir, _manifest(run_dir))

    assert str(error.value) == f"frozen manifest hash mismatch for {filename}"


def test_frozen_sample_hash_is_checked_before_candidate_hash(tmp_path: Path) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    manifest = _manifest(run_dir)
    manifest["outputs_sha256"]["frozen_sample.json"] = "0" * 64
    manifest["outputs_sha256"]["candidate_labels.csv"] = "0" * 64

    with pytest.raises(
        ValueError, match=r"^frozen manifest hash mismatch for frozen_sample\.json$"
    ):
        dspark_runner._verify_frozen_input_hashes(run_dir, manifest)


@pytest.mark.parametrize("filename", ["frozen_sample.json", "candidate_labels.csv"])
def test_frozen_input_without_a_recorded_hash_is_a_mismatch(tmp_path: Path, filename: str) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    manifest = _manifest(run_dir)
    del manifest["outputs_sha256"][filename]

    with pytest.raises(
        ValueError, match=rf"^frozen manifest hash mismatch for {re.escape(filename)}$"
    ):
        dspark_runner._verify_frozen_input_hashes(run_dir, manifest)


@pytest.mark.parametrize("outputs", [None, [], "sha", 7])
def test_frozen_manifest_outputs_must_be_a_mapping(tmp_path: Path, outputs: object) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    manifest = _manifest(run_dir)
    manifest["outputs_sha256"] = outputs

    with pytest.raises(ValueError, match=r"^frozen manifest has no outputs_sha256 object$"):
        dspark_runner._verify_frozen_input_hashes(run_dir, manifest)


def test_frozen_manifest_without_outputs_is_reported_as_missing_object(tmp_path: Path) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    manifest = _manifest(run_dir)
    del manifest["outputs_sha256"]

    with pytest.raises(ValueError, match=r"^frozen manifest has no outputs_sha256 object$"):
        dspark_runner._verify_frozen_input_hashes(run_dir, manifest)


# _verify_frozen_selection


def test_frozen_selection_accepts_manifest_values_that_match_the_sample() -> None:
    manifest = {"sample_ids_sha256": SAMPLE_IDS_SHA256, "seed": 42, "sample_size": 100}

    assert dspark_runner._verify_frozen_selection(manifest, {"selection": _selection()}) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [("sample_ids_sha256", "b" * 64), ("seed", 7), ("sample_size", 99)],
)
def test_each_frozen_selection_value_must_match_the_sample(key: str, value: object) -> None:
    manifest = {"sample_ids_sha256": SAMPLE_IDS_SHA256, "seed": 42, "sample_size": 100}
    manifest[key] = value

    with pytest.raises(
        ValueError, match=r"^frozen manifest does not match frozen_sample\.json$"
    ) as error:
        dspark_runner._verify_frozen_selection(manifest, {"selection": _selection()})

    assert str(error.value) == "frozen manifest does not match frozen_sample.json"


@pytest.mark.parametrize("key", ["sample_ids_sha256", "seed", "sample_size"])
def test_frozen_selection_requires_each_manifest_value(key: str) -> None:
    manifest = {"sample_ids_sha256": SAMPLE_IDS_SHA256, "seed": 42, "sample_size": 100}
    del manifest[key]

    with pytest.raises(ValueError, match=r"^frozen manifest does not match frozen_sample\.json$"):
        dspark_runner._verify_frozen_selection(manifest, {"selection": _selection()})


@pytest.mark.parametrize("key", ["sample_ids_sha256", "seed", "sample_size"])
def test_frozen_selection_requires_each_sample_value(key: str) -> None:
    manifest = {"sample_ids_sha256": SAMPLE_IDS_SHA256, "seed": 42, "sample_size": 100}
    selection = _selection()
    del selection[key]

    with pytest.raises(ValueError, match=r"^frozen manifest does not match frozen_sample\.json$"):
        dspark_runner._verify_frozen_selection(manifest, {"selection": selection})


# _validate_frozen_manifest


def test_validate_frozen_manifest_returns_the_manifest_file_digest(tmp_path: Path) -> None:
    run_dir, sample = _frozen_run(tmp_path)
    expected = _sha256_bytes((run_dir / "manifest.json").read_bytes())

    assert dspark_runner._validate_frozen_manifest(run_dir, sample) == expected


def test_validate_frozen_manifest_requires_manifest_json_by_name(tmp_path: Path) -> None:
    run_dir, sample = _frozen_run(tmp_path)
    (run_dir / "manifest.json").unlink()

    with pytest.raises(ValueError, match=rf"^{re.escape(MANIFEST_MISSING)}$"):
        dspark_runner._validate_frozen_manifest(run_dir, sample)


def test_validate_frozen_manifest_applies_the_selection_check(tmp_path: Path) -> None:
    run_dir, sample = _frozen_run(tmp_path)
    manifest = _manifest(run_dir)
    manifest["seed"] = 7
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^frozen manifest does not match frozen_sample\.json$"):
        dspark_runner._validate_frozen_manifest(run_dir, sample)


def test_validate_frozen_manifest_applies_the_input_hash_check(tmp_path: Path) -> None:
    run_dir, sample = _frozen_run(tmp_path)
    (run_dir / "candidate_labels.csv").write_bytes(b"changed\n")

    with pytest.raises(
        ValueError, match=r"^frozen manifest hash mismatch for candidate_labels\.csv$"
    ):
        dspark_runner._validate_frozen_manifest(run_dir, sample)


# _prediction_row


def test_unparsed_output_never_receives_a_candidate_name() -> None:
    row = {
        "sample_id": "s1",
        "sentence": "Dry grassland.",
        "eunis_code": "T11",
        "eunis_name": "Forest",
    }
    output = {"text": "</think>not a candidate", "meta_info": {"finish_reason": "stop"}}
    names: dict[Any, str] = {None: "Stray name", "T11": "Forest"}

    prediction = dspark_runner._prediction_row(row, 12, "prompt-hash", output, 0.5, ["T11"], names)

    assert prediction["parsed_eunis_code"] is None
    assert prediction["parsed_eunis_name"] is None
    assert prediction["correct_top1"] is None
    assert prediction["parse_status"] == "invalid"


def test_parsed_code_receives_its_candidate_name() -> None:
    row = {"sample_id": "s1", "sentence": "Forest.", "eunis_code": "T11", "eunis_name": "Forest"}
    output = {"text": "</think>T11", "meta_info": {"finish_reason": "stop"}}

    prediction = dspark_runner._prediction_row(
        row, 12, "prompt-hash", output, 0.5, ["T11"], {"T11": "Forest"}
    )

    assert prediction["parsed_eunis_code"] == "T11"
    assert prediction["parsed_eunis_name"] == "Forest"
    assert prediction["correct_top1"] is True


# sample_id_of


_ROW = {
    "source_pbf": "québec-latest.osm.pbf",
    "osm_type": "way",
    "osm_id": 7,
    "description_identity": "descripción-é",
    "tag_key": "description",
    "sentence_index": 0,
    "text_sha256": "f" * 64,
    "sentence": "Ignored sentence text.",
    "eunis_code": "T11",
    "language_code": "eng",
}

_EXPECTED_PAYLOAD = (
    '["québec-latest.osm.pbf","way",7,"descripción-é","description",0,"' + "f" * 64 + '"]'
)


def test_sample_id_is_sha256_of_the_compact_ordered_identity_json() -> None:
    assert sample_id_of(_ROW) == hashlib.sha256(_EXPECTED_PAYLOAD.encode("utf-8")).hexdigest()


def test_sample_id_is_pinned_for_the_identity_payload() -> None:
    assert sample_id_of(_ROW) == (
        "9b898a92eb8f66a3969f6d64757f71ae678d2e054fac808138fd3754be14da11"
    )


def test_sample_id_ignores_fields_outside_the_occurrence_identity() -> None:
    changed = {**_ROW, "sentence": "Different text.", "eunis_code": "E1", "language_code": "fra"}

    assert sample_id_of(changed) == sample_id_of(_ROW)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_pbf", "other-latest.osm.pbf"),
        ("osm_type", "node"),
        ("osm_id", 8),
        ("description_identity", "description-2"),
        ("tag_key", "name"),
        ("sentence_index", 1),
        ("text_sha256", "e" * 64),
    ],
)
def test_sample_id_changes_when_any_identity_field_changes(field: str, value: object) -> None:
    changed = {**_ROW, field: value}

    assert sample_id_of(changed) != sample_id_of(_ROW)


@pytest.mark.parametrize(
    "field",
    [
        "source_pbf",
        "osm_type",
        "osm_id",
        "description_identity",
        "tag_key",
        "sentence_index",
        "text_sha256",
    ],
)
def test_sample_id_requires_each_identity_field(field: str) -> None:
    row = {key: value for key, value in _ROW.items() if key != field}

    with pytest.raises(KeyError, match=rf"^'{field}'$"):
        sample_id_of(row)


# _leaves_groups_feasible and _take


def test_a_pick_that_leaves_its_own_group_one_cell_short_is_infeasible() -> None:
    available = {"yes": {"a"}, "no": {"x", "y"}}
    selected: dict[str, list[str]] = {"yes": [], "no": []}

    assert _leaves_groups_feasible("a", "yes", available, selected, 2) is False


def test_a_pick_that_leaves_every_group_enough_cells_is_feasible() -> None:
    available = {"yes": {"a", "b"}, "no": {"x", "y"}}
    selected: dict[str, list[str]] = {"yes": [], "no": []}

    assert _leaves_groups_feasible("a", "yes", available, selected, 2) is True


def test_take_records_the_chosen_cell_and_removes_it_from_both_groups() -> None:
    available = {"yes": {"a", "b"}, "no": {"a", "c"}}
    selected: dict[str, list[str]] = {"yes": [], "no": []}
    nearest: dict[str, float] = {}
    centres = {"a": (0.0, 0.0), "b": (0.0, 1.0), "c": (0.0, 2.0)}

    _take("a", "yes", available, selected, nearest, centres)

    assert available == {"yes": {"b"}, "no": {"c"}}
    assert selected == {"yes": ["a"], "no": []}
    assert nearest == {
        "b": _haversine_km((0.0, 1.0), (0.0, 0.0)),
        "c": _haversine_km((0.0, 2.0), (0.0, 0.0)),
    }
