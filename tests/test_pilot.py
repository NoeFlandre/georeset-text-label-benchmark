"""Tests for the deterministic EUNIS embedding pilot primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from georeset_text_label_benchmark.pilot import cli, runner
from georeset_text_label_benchmark.pilot.protocol import (
    CANDIDATE_LABELS_SHA256,
    OVERLAP_DATASET,
    OVERLAP_PARQUET_SHA256,
    OVERLAP_REVISION,
)
from georeset_text_label_benchmark.pilot.sampling import (
    _sample_id,
    build_candidate_labels,
    select_distinct_sample,
)


def _candidate_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pilot_data/eunis_candidate_labels.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pinned EUNIS candidate table is missing")


def _assert_run_file_names(run_dir: Path) -> None:
    assert {path.name for path in run_dir.iterdir()} == {
        "candidate_labels.csv",
        "frozen_sample.json",
        "manifest.json",
        "metrics.json",
        "predictions.parquet",
    }


def _assert_manifest_hash_path_names(calls: Any) -> None:
    names = [call.args[0].name for call in calls]
    assert {
        "frozen_sample.json",
        "candidate_labels.csv",
        "predictions.parquet",
        "metrics.json",
        "manifest.json",
    }.issubset(names)
    assert not {"MANIFEST.JSON", "CANDIDATE_LABELS.CSV"}.intersection(names)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_value_error(action: Any, expected: str) -> None:
    with pytest.raises(ValueError, match=re.escape(expected)) as error:
        action()
    assert str(error.value) == expected


def _row(
    osm_id: int,
    sentence: str,
    *,
    code: str = "T11",
    language: str | None = "eng",
) -> dict[str, Any]:
    return {
        "source_pbf": "test-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": osm_id,
        "description_identity": f"description-{osm_id}",
        "tag_key": "description",
        "sentence_index": 0,
        "sentence": sentence,
        "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
        "language_code": language,
        "eunis_code": code,
        "eunis_name": "Temperate forest",
    }


def _taxonomy(name: str, description: str) -> dict[str, str]:
    return {
        "name": name,
        "description": description,
        "classification_release": "terrestrial-2021",
        "classification_source": "EEA EUNIS terrestrial habitat classification",
        "source_sha256": "a" * 64,
        "license": "CC-BY-4.0",
    }


def _candidate_csv(path: Path, names: Mapping[str, str]) -> Path:
    fields = list(runner.CANDIDATE_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for code, name in sorted(names.items()):
            description = f"Definition for {code}."
            writer.writerow(
                {
                    "eunis_code": code,
                    "eunis_name": name,
                    "eunis_description": description,
                    "candidate_text": f"{name}\n{description}",
                    "classification_release": "terrestrial-2021",
                    "classification_source": "https://doi.org/test",
                    "source_sha256": "a" * 64,
                    "license": "CC-BY-4.0",
                }
            )
    return path


def _small_source(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "overlap.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _frozen_sample(run_dir: Path, candidate_csv: Path) -> list[dict[str, Any]]:
    with candidate_csv.open(encoding="utf-8", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    t11 = next(item for item in candidates if item["eunis_code"] == "T11")
    rows = [_row(index, f"pilot sentence {index}") for index in range(1, 4)]
    for row in rows:
        row["eunis_name"] = t11["eunis_name"]
    from georeset_text_label_benchmark.pilot.sampling import select_distinct_sample

    selected = select_distinct_sample(rows, size=3, seed=42)
    ids = [row["sample_id"] for row in selected]
    shutil.copyfile(candidate_csv, run_dir / "candidate_labels.csv")
    sample = {
        "format_version": 1,
        "source": {
            "dataset": OVERLAP_DATASET,
            "revision": OVERLAP_REVISION,
            "file": "overlap.parquet",
            "sha256": OVERLAP_PARQUET_SHA256,
            "license": "OpenStreetMap ODbL-1.0; European Environment Agency CC-BY-4.0",
        },
        "source_coverage": {
            "row_count": 224789,
            "unique_sentence_hash_count": 137464,
            "unique_polygon_count": 200941,
            "observed_eunis_class_count": 158,
        },
        "selection": {
            "method": (
                "Sort occurrences by SHA256(seed + ':' + stable occurrence ID), then retain "
                "the first rows with unseen exact text hashes and unseen polygon keys."
            ),
            "seed": 42,
            "sample_size": len(selected),
            "sample_ids_sha256": runner._sha256_json(ids),
            "unique_sentence_hash_count": len({row["text_sha256"] for row in selected}),
            "unique_polygon_count": len(
                {(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in selected}
            ),
        },
        "candidate_labels": {
            "file": "candidate_labels.csv",
            "sha256": runner.sha256_file(run_dir / "candidate_labels.csv"),
            "count": len(candidates),
            "codes": [item["eunis_code"] for item in candidates],
            "text_method": "Exact pinned EUNIS English name, newline, exact EEA description.",
        },
        "selected_rows": selected,
    }
    runner._write_json_exclusive(run_dir / "frozen_sample.json", sample)
    return selected


def _reverse_frozen_rows(payload: dict[str, Any]) -> None:
    payload["selected_rows"].reverse()
    payload["selection"]["sample_ids_sha256"] = runner._sha256_json(
        [row["sample_id"] for row in payload["selected_rows"]]
    )


def test_sample_is_stable_order_independent_and_occurrence_deduplicated() -> None:
    rows = [_row(i, f"sentence {i}") for i in range(1, 6)]
    rows.append(dict(rows[0]))

    first = select_distinct_sample(rows, size=3, seed=42)
    second = select_distinct_sample(list(reversed(rows)), size=3, seed=42)

    assert first == second
    assert [row["sample_id"] for row in first] == [
        "c4b1b52fb1c2c0ec831354f1a9cd4904e6757faf4bbf721addea40a6e9599bfe",
        "aae199d39d84009b73f5697cbd16394eb3b88c1d09aaccdec1e73b2ee3b14a0a",
        "4c3bd7d1799aa1b8403cdb3fa5b894d7b3cadb2066b76d31c25dbbb354742ada",
    ]
    assert len(first) == 3
    assert len({row["sample_id"] for row in first}) == 3
    assert len({row["text_sha256"] for row in first}) == 3
    assert len({(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in first}) == 3
    assert all(row["sentence"].startswith("sentence ") for row in first)


def test_sample_identity_hash_uses_utf8_without_ascii_escaping() -> None:
    row = _row(1, "sentence")
    row["description_identity"] = "forêt"

    assert _sample_id(row) == "e5be146774b05d51421ffff91f6c363c91296924dc8d51c568885d72de44dcee"


def test_ranked_sample_skips_repeated_text_and_polygon_occurrences() -> None:
    rows = {
        "first": _row(1, "repeat"),
        "same-text": _row(2, "repeat"),
        "same-polygon": _row(1, "different"),
        "fourth": _row(3, "fourth"),
        "fifth": _row(4, "fifth"),
    }

    selected = runner.select_distinct_sample(rows.values(), size=3, seed=42)

    assert [row["sample_id"] for row in selected] == [
        "89eb72c331c318479355ab8a55ca2cb4ca0304115a56fa38bd5c101f39b6d215",
        "0de7d9df5bb9b7571416f3929247223b93e8fba45d917a9fe4e55e98425fd6f2",
        "57bcf434153aff262fb51ae2b40d443a3eadb3f2eb9267009353836abb274d29",
    ]
    assert len({row["text_sha256"] for row in selected}) == 3
    assert len({(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in selected}) == 3


@pytest.mark.parametrize(
    "rows",
    [
        [_row(1, "repeat"), _row(2, "repeat")],
        [_row(1, "first"), _row(1, "second")],
    ],
)
def test_sample_rejects_insufficient_distinct_sentences_or_polygons(
    rows: list[dict[str, Any]],
) -> None:
    with pytest.raises(
        ValueError, match="cannot select 2 rows with distinct sentence hashes and polygons"
    ):
        select_distinct_sample(rows, size=2, seed=42)


def test_sample_rejects_sentence_hash_mismatch_and_missing_join_keys() -> None:
    bad_hash = _row(1, "sentence")
    bad_hash["text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sentence hash mismatch"):
        select_distinct_sample([bad_hash], size=1, seed=42)

    bad_key = _row(2, "another sentence")
    bad_key["osm_id"] = None
    with pytest.raises(ValueError, match="osm_id must be a non-null integer"):
        select_distinct_sample([bad_key], size=1, seed=42)


@pytest.mark.parametrize(
    "field",
    [
        "source_pbf",
        "osm_type",
        "description_identity",
        "tag_key",
        "sentence",
        "text_sha256",
        "eunis_code",
        "eunis_name",
    ],
)
def test_sample_rejects_each_missing_required_string(field: str) -> None:
    row = _row(1, "sentence")
    row[field] = ""
    with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
        select_distinct_sample([row], size=1, seed=42)


@pytest.mark.parametrize(
    ("field", "value"),
    [("osm_id", True), ("osm_id", -1), ("sentence_index", True), ("sentence_index", -1)],
)
def test_sample_rejects_boolean_and_negative_integer_keys(field: str, value: Any) -> None:
    row = _row(1, "sentence")
    row[field] = value
    message = (
        "non-null integer" if isinstance(value, bool) else "must be positive and sentence indices"
    )
    with pytest.raises(ValueError, match=message):
        select_distinct_sample([row], size=1, seed=42)


def test_sample_rejects_bad_values_and_conflicting_duplicate_occurrences() -> None:
    empty_sentence = _row(1, "")
    with pytest.raises(ValueError, match="sentence must be a non-empty string"):
        select_distinct_sample([empty_sentence], size=1, seed=42)
    bad_language = _row(2, "sentence")
    bad_language["language_code"] = 7
    with pytest.raises(ValueError, match="language_code must be"):
        select_distinct_sample([bad_language], size=1, seed=42)
    bad_id = _row(3, "sentence")
    bad_id["osm_id"] = 0
    with pytest.raises(ValueError, match="OSM IDs must be positive"):
        select_distinct_sample([bad_id], size=1, seed=42)
    first = _row(4, "same occurrence")
    conflict = dict(first, eunis_code="T12", eunis_name="Other habitat")
    with pytest.raises(ValueError, match="conflicting duplicate occurrence"):
        select_distinct_sample([first, conflict], size=1, seed=42)
    with pytest.raises(ValueError, match="size must be a positive integer"):
        select_distinct_sample([first], size=0, seed=42)
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        select_distinct_sample([first], size=1, seed=-1)
    with pytest.raises(ValueError, match="size must be a positive integer"):
        select_distinct_sample([first], size=True, seed=1)
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        select_distinct_sample([first], size=1, seed=True)
    assert len(select_distinct_sample([first], size=1, seed=0)) == 1


def test_sampling_validation_errors_are_exact() -> None:
    from georeset_text_label_benchmark.pilot import sampling

    bad_language = _row(1, "sentence")
    bad_language["language_code"] = 7
    bad_hash = _row(2, "sentence")
    bad_hash["text_sha256"] = "0" * 64
    bad_id = _row(0, "sentence")
    bad_id["osm_id"] = 0
    bad_index = _row(3, "sentence")
    bad_index["sentence_index"] = -1
    _assert_value_error(
        lambda: sampling._validate_language(7), "language_code must be a non-empty string or null"
    )
    _assert_value_error(
        lambda: sampling._validate_sentence_hash("sentence", "0" * 64),
        "sentence hash mismatch",
    )
    _assert_value_error(
        lambda: sampling._validate_row(bad_id),
        "OSM IDs must be positive and sentence indices non-negative",
    )
    _assert_value_error(
        lambda: sampling._validate_row(bad_index),
        "OSM IDs must be positive and sentence indices non-negative",
    )
    _assert_value_error(
        lambda: sampling._validate_sample_size(0), "size must be a positive integer"
    )
    _assert_value_error(
        lambda: sampling._validate_sample_seed(-1), "seed must be a non-negative integer"
    )
    _assert_value_error(
        lambda: sampling.build_candidate_labels([_row(4, "four")], {}),
        "taxonomy codes do not match observed EUNIS codes",
    )
    _assert_value_error(
        lambda: sampling._validate_candidate_source_hash("T11", "X" * 64),
        "source_sha256 must be a lowercase SHA-256 for T11",
    )
    candidate_details = _taxonomy("Name", "A definition.")
    candidate_details["source_sha256"] = "X" * 64
    _assert_value_error(
        lambda: sampling._candidate_record("T11", "Name", candidate_details),
        "source_sha256 must be a lowercase SHA-256 for T11",
    )
    _assert_value_error(
        lambda: sampling._validate_row(bad_language),
        "language_code must be a non-empty string or null",
    )
    _assert_value_error(lambda: sampling._validate_row(bad_hash), "sentence hash mismatch")


def test_candidate_record_retains_all_provenance_fields() -> None:
    taxonomy = runner._read_taxonomy(_candidate_file())
    records = runner._candidate_records(taxonomy)

    assert len(records) == 158
    assert all(
        set(record)
        == {
            "eunis_code",
            "eunis_name",
            "candidate_text",
            "classification_release",
            "classification_source",
            "source_sha256",
            "license",
        }
        for record in records
    )


def test_candidate_text_uses_exact_source_names_and_authoritative_descriptions() -> None:
    rows = [_row(1, "one"), _row(2, "two", code="MA221")]
    rows[1]["eunis_name"] = "Atlantic saltmarsh driftlines"
    taxonomy: Mapping[str, Mapping[str, str]] = {
        "T11": _taxonomy("Temperate forest", "Forest description."),
        "MA221": {
            **_taxonomy("Atlantic saltmarsh driftlines", "Marine description."),
            "classification_release": "marine-2022",
            "classification_source": "EEA EUNIS marine habitat classification",
        },
    }

    candidates = build_candidate_labels(rows, taxonomy)

    assert candidates == [
        {
            "eunis_code": "MA221",
            "eunis_name": "Atlantic saltmarsh driftlines",
            "eunis_description": "Marine description.",
            "candidate_text": "Atlantic saltmarsh driftlines\nMarine description.",
            "classification_release": "marine-2022",
            "classification_source": "EEA EUNIS marine habitat classification",
            "source_sha256": "a" * 64,
            "license": "CC-BY-4.0",
        },
        {
            "eunis_code": "T11",
            "eunis_name": "Temperate forest",
            "eunis_description": "Forest description.",
            "candidate_text": "Temperate forest\nForest description.",
            "classification_release": "terrestrial-2021",
            "classification_source": "EEA EUNIS terrestrial habitat classification",
            "source_sha256": "a" * 64,
            "license": "CC-BY-4.0",
        },
    ]


def test_candidate_text_fails_closed_on_code_name_or_definition_gaps() -> None:
    row = _row(1, "one")
    with pytest.raises(ValueError, match="taxonomy codes do not match observed EUNIS codes"):
        build_candidate_labels([row], {})

    mismatch = _taxonomy("Different forest name", "Forest description.")
    with pytest.raises(ValueError, match="EUNIS name mismatch for T11"):
        build_candidate_labels([row], {"T11": mismatch})

    empty = _taxonomy("Temperate forest", " ")
    with pytest.raises(ValueError, match="description must be non-empty for T11"):
        build_candidate_labels([row], {"T11": empty})

    bad_checksum = _taxonomy("Temperate forest", "Forest description.")
    bad_checksum["source_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="source_sha256 must be a lowercase SHA-256"):
        build_candidate_labels([row], {"T11": bad_checksum})


def test_candidate_labels_reject_inconsistent_names_for_one_code() -> None:
    first = _row(1, "one")
    second = _row(2, "two")
    second["eunis_name"] = "Other"
    with pytest.raises(ValueError, match="inconsistent EUNIS name for T11"):
        build_candidate_labels(
            [first, second], {"T11": _taxonomy("Temperate forest", "Forest description.")}
        )


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.options: list[dict[str, Any]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(texts)
        self.options.append(kwargs)
        return {
            "input_ids": torch.ones((len(texts), 2), dtype=torch.int64),
            "attention_mask": torch.ones((len(texts), 2), dtype=torch.int64),
        }


class _RecordingModel:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> _RecordingModel:
        self.eval_called = True
        return self

    def __call__(self, **batch: torch.Tensor) -> SimpleNamespace:
        count, tokens = batch["input_ids"].shape
        hidden = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]]).repeat(count, 1, 1)
        assert tokens == 2
        return SimpleNamespace(last_hidden_state=hidden)


def test_checked_in_candidate_file_has_158_cc_by_definitions() -> None:
    path = _candidate_file()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert runner.sha256_file(path) == CANDIDATE_LABELS_SHA256
    assert len(rows) == 158
    assert [row["eunis_code"] for row in rows] == sorted({row["eunis_code"] for row in rows})
    assert all(row["eunis_description"].strip() for row in rows)
    assert all(
        row["candidate_text"] == f"{row['eunis_name']}\n{row['eunis_description']}" for row in rows
    )
    assert {row["license"] for row in rows} == {"CC-BY-4.0"}
    assert len({row["source_sha256"] for row in rows}) == 2


def test_freeze_sample_writes_pinned_rows_without_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(index, f"row {index}") for index in range(1, 5)]
    source = _small_source(tmp_path, rows)
    candidates = _candidate_csv(tmp_path / "candidates.csv", {"T11": "Temperate forest"})
    monkeypatch.setattr(runner, "EXPECTED_SOURCE_ROWS", len(rows))
    monkeypatch.setattr(runner, "EXPECTED_CANDIDATES", 1)
    output = tmp_path / "nested" / "frozen"
    observed_hash_paths: list[str] = []
    original_sha256_file = runner.sha256_file

    def recording_sha256_file(path: Path) -> str:
        observed_hash_paths.append(path.name)
        return original_sha256_file(path)

    monkeypatch.setattr(runner, "sha256_file", recording_sha256_file)

    result = runner.freeze_sample(
        source,
        candidates,
        output,
        expected_source_sha256=runner.sha256_file(source),
        expected_candidate_sha256=None,
        size=3,
        seed=42,
    )

    payload = json.loads((output / "frozen_sample.json").read_text(encoding="utf-8"))
    expected_ids = [
        "ae2397129e3c13e30d26edebc43fe433cabef1b98352cb23801ba8efa26b6e4e",
        "f3a6316dca4121004fe9c944a291fa0cd59c01fa29d09a6251c0ecfc780f86fe",
        "587186f31154dc70f0a56037cd65d2b8d606f52d5fdf79cc53fd177e4e026845",
    ]
    expected_rows = [
        {**rows[1], "sample_id": expected_ids[0]},
        {**rows[3], "sample_id": expected_ids[1]},
        {**rows[2], "sample_id": expected_ids[2]},
    ]
    sample_ids_sha256 = "f65ae38657a83dc1202551ce0b4c998a53da06cbc009d3c8704d59b76d439950"
    expected = {
        "format_version": 1,
        "source": {
            "dataset": OVERLAP_DATASET,
            "revision": OVERLAP_REVISION,
            "file": "overlap.parquet",
            "sha256": _file_sha256(source),
            "license": "OpenStreetMap ODbL-1.0; European Environment Agency CC-BY-4.0",
        },
        "source_coverage": {
            "row_count": 4,
            "unique_sentence_hash_count": 4,
            "unique_polygon_count": 4,
            "observed_eunis_class_count": 1,
        },
        "selection": {
            "method": (
                "Sort occurrences by SHA256(seed + ':' + stable occurrence ID), then retain "
                "the first rows with unseen exact text hashes and unseen polygon keys."
            ),
            "seed": 42,
            "sample_size": 3,
            "sample_ids_sha256": sample_ids_sha256,
            "unique_sentence_hash_count": 3,
            "unique_polygon_count": 3,
        },
        "candidate_labels": {
            "file": "candidate_labels.csv",
            "sha256": _file_sha256(candidates),
            "count": 1,
            "codes": ["T11"],
            "text_method": "Exact pinned EUNIS English name, newline, exact EEA description.",
        },
        "selected_rows": expected_rows,
    }
    assert payload == expected
    assert (output / "frozen_sample.json").read_bytes() == (
        json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert result == {
        "output_dir": str(output),
        "sample_size": 3,
        "candidate_count": 1,
        "sample_ids_sha256": sample_ids_sha256,
        "frozen_sample_sha256": _file_sha256(output / "frozen_sample.json"),
    }
    assert (output / "candidate_labels.csv").read_bytes() == candidates.read_bytes()
    assert {path.name for path in output.iterdir()} == {
        "candidate_labels.csv",
        "frozen_sample.json",
    }
    assert observed_hash_paths.count("frozen_sample.json") == 1
    assert not (output / "predictions.parquet").exists()
    with pytest.raises(FileExistsError, match="pilot directory already exists"):
        runner.freeze_sample(
            source, candidates, output, expected_source_sha256=runner.sha256_file(source)
        )


def test_freeze_sample_rejects_directory_created_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _small_source(tmp_path, [_row(1, "one")])
    candidates = _candidate_csv(tmp_path / "candidates.csv", {"T11": "Temperate forest"})
    monkeypatch.setattr(runner, "EXPECTED_SOURCE_ROWS", 1)
    monkeypatch.setattr(runner, "EXPECTED_CANDIDATES", 1)
    output = tmp_path / "racing-parent" / "pilot"
    original_mkdir = Path.mkdir

    def racing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == output:
            original_mkdir(output.parent, parents=True, exist_ok=True)
            original_mkdir(output, parents=True, exist_ok=True)
        original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", racing_mkdir)
    with pytest.raises(FileExistsError):
        runner.freeze_sample(
            source,
            candidates,
            output,
            expected_source_sha256=runner.sha256_file(source),
            expected_candidate_sha256=None,
            size=1,
            seed=42,
        )


def test_freeze_sample_rejects_unpinned_inputs_and_short_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _small_source(tmp_path, [_row(1, "one")])
    candidates = _candidate_csv(tmp_path / "candidates.csv", {"T11": "Temperate forest"})
    actual = runner.sha256_file(source)

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        runner.freeze_sample(
            source, candidates, tmp_path / "bad-hash", expected_source_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="pinned overlap snapshot"):
        runner.freeze_sample(source, candidates, tmp_path / "bad-revision", source_revision="bad")
    with pytest.raises(ValueError, match="candidate CSV SHA-256"):
        runner.freeze_sample(
            source,
            candidates,
            tmp_path / "bad-candidate",
            expected_source_sha256=actual,
            expected_candidate_sha256="0" * 64,
        )
    monkeypatch.setattr(runner, "EXPECTED_SOURCE_ROWS", 2)
    with pytest.raises(ValueError, match="expected 2 overlap rows, got 1"):
        runner.freeze_sample(
            source,
            candidates,
            tmp_path / "short",
            expected_source_sha256=actual,
            expected_candidate_sha256=runner.sha256_file(candidates),
        )


def test_freeze_sample_rejects_wrong_candidate_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _small_source(tmp_path, [_row(1, "one")])
    candidates = _candidate_csv(tmp_path / "candidates.csv", {"T11": "Temperate forest"})
    monkeypatch.setattr(runner, "EXPECTED_SOURCE_ROWS", 1)
    monkeypatch.setattr(runner, "EXPECTED_CANDIDATES", 2)
    with pytest.raises(ValueError, match="expected 2 candidate classes"):
        runner.freeze_sample(
            source,
            candidates,
            tmp_path / "wrong-candidate-count",
            expected_source_sha256=runner.sha256_file(source),
            expected_candidate_sha256=runner.sha256_file(candidates),
        )


def test_taxonomy_csv_validation_rejects_schema_duplicates_and_text_changes(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "bad-schema.csv"
    malformed.write_text("code\nT11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate CSV schema"):
        runner._read_taxonomy(malformed)

    valid = _candidate_csv(tmp_path / "valid.csv", {"T11": "Temperate forest"})
    with valid.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with valid.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.CANDIDATE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([rows[0], rows[0]])
    with pytest.raises(ValueError, match="missing or duplicate candidate code"):
        runner._read_taxonomy(valid)

    rows[0]["candidate_text"] = "altered text"
    with valid.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.CANDIDATE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(rows[0])
    with pytest.raises(ValueError, match="candidate text mismatch for T11"):
        runner._read_taxonomy(valid)


def test_taxonomy_reader_preserves_unicode_and_embedded_crlf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unicode.csv"
    description = "First line\r\nSecond line"
    row = {
        "eunis_code": "T11",
        "eunis_name": "Forêt tempérée",
        "eunis_description": description,
        "candidate_text": f"Forêt tempérée\n{description}",
        "classification_release": "terrestrial-2021",
        "classification_source": "https://doi.org/test",
        "source_sha256": "a" * 64,
        "license": "CC-BY-4.0",
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.CANDIDATE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    observed: list[tuple[str | None, str | None]] = []
    original_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == path:
            observed.append((kwargs.get("encoding"), kwargs.get("newline")))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    taxonomy = runner._read_taxonomy(path)

    assert observed == [("utf-8", "")]
    assert taxonomy["T11"]["name"] == "Forêt tempérée"
    assert taxonomy["T11"]["description"] == description


def test_json_writers_are_exclusive_utf8_and_preserve_unicode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    observed: list[tuple[str | None, str | None, str | None]] = []
    original_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == path:
            observed.append(
                (args[0] if args else None, kwargs.get("encoding"), kwargs.get("newline"))
            )
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    runner._write_json_exclusive(path, {"name": "Forêt"})

    assert observed == [("x", "utf-8", "\n")]
    assert path.read_bytes() == '{\n  "name": "Forêt"\n}\n'.encode()
    with pytest.raises(FileExistsError):
        runner._write_json_exclusive(path, {"name": "other"})


def test_json_hash_uses_canonical_utf8_for_unicode() -> None:
    expected = hashlib.sha256('["forêt"]'.encode()).hexdigest()

    assert runner._sha256_json(["forêt"]) == expected
    assert runner._sha256_json({"z": 1, "a": 2}) == (
        "c2985c5ba6f7d2a55e768f92490ca09388e95bc4cccb9fdf11b15f4d42f93e73"
    )


def test_source_reader_projects_only_pinned_columns_in_bounded_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _small_source(tmp_path, [_row(1, "one")])
    original = pq.ParquetFile
    observed: list[dict[str, Any]] = []

    class RecordingParquetFile:
        def __init__(self, path: Path) -> None:
            self.parquet = original(path)
            self.metadata = self.parquet.metadata

        def iter_batches(self, **kwargs: Any) -> Any:
            observed.append(kwargs)
            return self.parquet.iter_batches(**kwargs)

    monkeypatch.setattr(runner.pq, "ParquetFile", RecordingParquetFile)
    rows, row_count = runner._read_source_rows(source)

    assert row_count == 1
    assert rows[0]["sentence"] == "one"
    assert observed == [{"columns": list(runner.SOURCE_COLUMNS), "batch_size": 16_384}]


def test_runner_rejects_tampered_frozen_sample(tmp_path: Path) -> None:
    candidates = _candidate_file()
    run_dir = tmp_path / "tampered"
    run_dir.mkdir()
    selected = _frozen_sample(run_dir, candidates)
    path = run_dir / "frozen_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"]["sample_ids_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sample ID checksum mismatch"):
        runner._read_frozen_sample(run_dir)

    payload["selected_rows"] = selected


def test_frozen_sample_reader_uses_exact_filename_and_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "read-sample"
    run_dir.mkdir()
    _frozen_sample(run_dir, _candidate_file())
    observed: list[tuple[str, str | None]] = []
    original_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.parent == run_dir:
            observed.append((self.name, kwargs.get("encoding")))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    sample, rows = runner._read_frozen_sample(run_dir)

    assert len(rows) == sample["selection"]["sample_size"] == 3
    assert observed[0][0] == "frozen_sample.json"
    assert observed[0][1] is not None
    assert observed[0][1].lower() == "utf-8"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dataset", "different overlap dataset revision"),
        ("revision", "different overlap dataset revision"),
        ("source_hash", "different overlap Parquet hash"),
        ("row_count", "row count does not match"),
        ("duplicate", "unique IDs, sentence hashes, and polygons"),
        ("checksum", "sample ID checksum mismatch"),
        ("order", "deterministic sampling protocol"),
    ],
)
def test_frozen_sample_validation_rejects_each_contract_mismatch(
    tmp_path: Path, mutation: str, message: str
) -> None:
    candidates = _candidate_file()
    run_dir = tmp_path / mutation
    run_dir.mkdir()
    selected = _frozen_sample(run_dir, candidates)
    path = run_dir / "frozen_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutators = {
        "dataset": lambda data: data["source"].update(dataset="other/dataset"),
        "revision": lambda data: data["source"].update(revision="f" * 40),
        "source_hash": lambda data: data["source"].update(sha256="0" * 64),
        "row_count": lambda data: data["selection"].update(sample_size=4),
        "duplicate": lambda data: data["selected_rows"].__setitem__(1, selected[0]),
        "checksum": lambda data: data["selection"].update(sample_ids_sha256="0" * 64),
        "order": _reverse_frozen_rows,
    }
    mutators[mutation](payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runner._read_frozen_sample(run_dir)


@pytest.mark.parametrize("mutation", ["checksum", "codes", "gold_name", "class_count"])
def test_candidate_rows_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    candidates = _candidate_file()
    run_dir = tmp_path / mutation
    run_dir.mkdir()
    _frozen_sample(run_dir, candidates)
    frozen_path = run_dir / "frozen_sample.json"
    sample = json.loads(frozen_path.read_text(encoding="utf-8"))
    if mutation == "checksum":
        (run_dir / "candidate_labels.csv").write_text("corrupt", encoding="utf-8")
    elif mutation == "codes":
        sample["candidate_labels"]["codes"][0] = "wrong"
        frozen_path.write_text(json.dumps(sample), encoding="utf-8")
    elif mutation == "gold_name":
        sample["selected_rows"][0]["eunis_name"] = "Wrong name"
        frozen_path.write_text(json.dumps(sample), encoding="utf-8")
    else:
        monkeypatch.setattr(runner, "EXPECTED_CANDIDATES", 159)
    expected = {
        "checksum": "candidate label file checksum mismatch",
        "codes": "candidate code list does not match",
        "gold_name": "frozen gold name mismatch",
        "class_count": "expected 159 candidate classes",
    }[mutation]
    with pytest.raises(ValueError, match=expected):
        runner._candidate_rows(run_dir, sample)


def _inject_e5_output_failure(monkeypatch: pytest.MonkeyPatch, failing_name: str) -> None:
    if failing_name == "predictions.parquet":
        write_table = runner.pq.write_table

        def write_then_fail(table: Any, path: Path, **kwargs: Any) -> None:
            write_table(table, path, **kwargs)
            raise OSError("injected publication failure")

        monkeypatch.setattr(runner.pq, "write_table", write_then_fail)
        return

    write_json = runner._write_json_exclusive

    def write_then_fail(path: Path, payload: Mapping[str, Any]) -> None:
        write_json(path, payload)
        if path.name == failing_name:
            raise OSError("injected publication failure")

    monkeypatch.setattr(runner, "_write_json_exclusive", write_then_fail)


def test_staged_hash_validation_reports_input_and_output_context(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    input_path = tmp_path / "frozen_sample.json"
    output_path = staging / "predictions.parquet"
    input_path.write_bytes(b"frozen")
    output_path.write_bytes(b"prediction")
    input_paths = {"frozen_sample.json": input_path}
    output_paths = {"predictions.parquet": output_path}

    for hashes, expected in (
        (
            {
                "frozen_sample.json": "0" * 64,
                "predictions.parquet": runner.sha256_file(output_path),
            },
            f"retained E5 staging input hash mismatch for frozen_sample.json: {staging}",
        ),
        (
            {
                "frozen_sample.json": runner.sha256_file(input_path),
                "predictions.parquet": "0" * 64,
            },
            f"retained E5 staging output hash mismatch for predictions.parquet: {staging}",
        ),
    ):
        with pytest.raises(ValueError, match="retained E5 staging") as error:
            runner._validate_staged_hashes(
                staging,
                {"outputs_sha256": hashes},
                input_paths,
                output_paths,
                "E5",
            )
        assert str(error.value) == expected


def test_staged_hash_validation_requires_a_hash_mapping(tmp_path: Path) -> None:
    staging = tmp_path / "staging"

    with pytest.raises(ValueError, match="has no output hashes") as error:
        runner._validate_staged_hashes(staging, {}, {}, {}, "E5")

    assert str(error.value) == f"retained E5 staging manifest has no output hashes: {staging}"


def _identity_embedding_vectors(
    selected: list[dict[str, Any]], candidate_rows: list[dict[str, str]]
) -> tuple[list[str], dict[str, int], dict[int, torch.Tensor]]:
    expected_codes = sorted(row["eunis_code"] for row in candidate_rows)[: len(selected)]
    basis = torch.eye(len(selected))
    vectors_by_text: dict[str, torch.Tensor] = {
        f"query: {row['sentence']}": basis[index] for index, row in enumerate(selected)
    }
    candidate_vectors = {code: basis[index] for index, code in enumerate(expected_codes)}
    vectors_by_text.update(
        {
            f"passage: {row['candidate_text']}": candidate_vectors.get(
                row["eunis_code"], torch.ones(len(selected))
            )
            for row in candidate_rows
        }
    )
    token_ids = {text: index for index, text in enumerate(vectors_by_text, start=1)}
    vectors_by_id = {token_ids[text]: vector for text, vector in vectors_by_text.items()}
    return expected_codes, token_ids, vectors_by_id


def _identity_expected_predictions(
    selected: list[dict[str, Any]], expected_codes: list[str]
) -> list[tuple[str, str, str, float]]:
    return [
        (row["sample_id"], row["sentence"], expected_codes[index], 1.0)
        for index, row in enumerate(selected)
    ]


def test_cli_parser_help_is_stable_and_defaults_are_pinned() -> None:
    parser = cli._parser()
    assert parser.format_help() == (
        "usage: georeset-pilot [-h] {freeze,run-dspark,run-dspark-smoke} ...\n\n"
        "positional arguments:\n"
        "  {freeze,run-dspark,run-dspark-smoke}\n"
        "    freeze              freeze selected rows before inference\n"
        "    run-dspark          predict one EUNIS code per row with pinned LFM2.5 +\n"
        "                        DSpark\n"
        "    run-dspark-smoke    run a bounded eight-row LFM2.5 + DSpark readiness\n"
        "                        smoke\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
    )
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["freeze"])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["run-dspark"])

    frozen = parser.parse_args(["freeze", "--source-parquet", "overlap.parquet"])
    assert frozen.source_parquet == Path("overlap.parquet")
    assert frozen.candidate_csv == Path("pilot_data/eunis_candidate_labels.csv")
    assert frozen.output_dir == Path("artifacts/pilot-100-seed42")
    assert frozen.sample_size == 100
    assert frozen.seed == 42
    dspark_running = parser.parse_args(
        ["run-dspark", "--computation-commit", "a" * 40, "--validation-commit", "b" * 40]
    )
    assert dspark_running.run_dir == Path("artifacts/pilot-100-seed42")
    assert dspark_running.output_dir is None
    assert dspark_running.model_cache == Path(".cache/model-dspark")

    explicit = parser.parse_args(
        [
            "freeze",
            "--source-parquet",
            "source.parquet",
            "--candidate-csv",
            "custom.csv",
            "--output-dir",
            "custom-output",
            "--sample-size",
            "7",
            "--seed",
            "0",
        ]
    )
    assert explicit.candidate_csv == Path("custom.csv")
    assert explicit.output_dir == Path("custom-output")
    assert explicit.sample_size == 7
    assert explicit.seed == 0


def test_cli_dispatches_exact_freeze_settings_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def freeze(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(("freeze", args, kwargs))
        return {"z": "é", "a": 1}

    monkeypatch.setattr(cli, "freeze_sample", freeze)
    assert cli.main(["freeze", "--source-parquet", "overlap.parquet"]) == 0
    assert capsys.readouterr().out == '{\n  "a": 1,\n  "z": "é"\n}\n'
    assert calls[-1] == (
        "freeze",
        (
            Path("overlap.parquet"),
            Path("pilot_data/eunis_candidate_labels.csv"),
            Path("artifacts/pilot-100-seed42"),
        ),
        {
            "expected_source_sha256": OVERLAP_PARQUET_SHA256,
            "source_revision": OVERLAP_REVISION,
            "size": 100,
            "seed": 42,
        },
    )
