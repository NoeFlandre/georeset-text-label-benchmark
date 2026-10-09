"""Tests for the deterministic EUNIS embedding pilot primitives."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from georeset_text_label_benchmark.pilot import runner
from georeset_text_label_benchmark.pilot.protocol import (
    CANDIDATE_LABELS_SHA256,
)
from georeset_text_label_benchmark.pilot.sampling import (
    build_candidate_labels,
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


def _reverse_frozen_rows(payload: dict[str, Any]) -> None:
    payload["selected_rows"].reverse()
    payload["selection"]["sample_ids_sha256"] = runner._sha256_json(
        [row["sample_id"] for row in payload["selected_rows"]]
    )


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
