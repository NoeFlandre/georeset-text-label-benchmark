"""Tests for the deterministic EUNIS embedding pilot primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from georeset_text_label_benchmark.pilot import cli, runner
from georeset_text_label_benchmark.pilot.embeddings import (
    average_pool,
    encode_texts,
    rank_candidates,
)
from georeset_text_label_benchmark.pilot.metrics import (
    _accuracy_counts,
    _true_positives,
    class_breakdown,
    classification_report,
    group_breakdown,
)
from georeset_text_label_benchmark.pilot.protocol import (
    CANDIDATE_LABELS_SHA256,
    OVERLAP_DATASET,
    OVERLAP_PARQUET_SHA256,
    OVERLAP_REVISION,
)
from georeset_text_label_benchmark.pilot.runner import MODEL_FILES
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


def test_metrics_include_all_candidate_classes_in_macro_f1() -> None:
    gold = ["A", "A", "B"]
    top1 = ["A", "B", "B"]
    top5 = [["A", "B", "C"], ["B", "A", "C"], ["B", "A", "C"]]

    report = classification_report(gold, top1, top5, ["A", "B", "C"])

    assert report == {
        "sample_count": 3,
        "candidate_class_count": 3,
        "top1_accuracy": pytest.approx(2 / 3),
        "top5_accuracy": pytest.approx(1.0),
        "macro_f1_all_candidates": pytest.approx(4 / 9),
        "macro_f1_definition": (
            "Unweighted mean of per-code F1 over all candidate codes; codes with no gold "
            "support and no predictions contribute zero."
        ),
    }
    classes = class_breakdown(gold, top1, top5, ["A", "B", "C"])
    assert classes == [
        {
            "eunis_code": "A",
            "support": 2,
            "top1_correct": 1,
            "top1_accuracy": 0.5,
            "top5_hits": 2,
            "top5_recall": 1.0,
        },
        {
            "eunis_code": "B",
            "support": 1,
            "top1_correct": 1,
            "top1_accuracy": 1.0,
            "top5_hits": 1,
            "top5_recall": 1.0,
        },
        {
            "eunis_code": "C",
            "support": 0,
            "top1_correct": 0,
            "top1_accuracy": None,
            "top5_hits": 0,
            "top5_recall": None,
        },
    ]


def test_metrics_reject_misaligned_or_unknown_predictions() -> None:
    with pytest.raises(ValueError, match="gold, top1, and top5 lengths must agree"):
        classification_report(["A"], [], [], ["A"])
    with pytest.raises(ValueError, match="unknown gold code: Z"):
        classification_report(["Z"], ["A"], [["A"]], ["A"])
    with pytest.raises(ValueError, match="top1 prediction must be first in top5"):
        classification_report(["A"], ["B"], [["A", "B"]], ["A", "B"])
    with pytest.raises(ValueError, match="unknown prediction code: Z"):
        classification_report(["A"], ["Z"], [["Z"]], ["A"])
    with pytest.raises(ValueError, match="unknown prediction code: Z"):
        classification_report(["A"], ["A"], [["A", "Z"]], ["A"])
    with pytest.raises(ValueError, match="gold, top1, and top5 lengths"):
        classification_report([], [], [], ["A"])
    with pytest.raises(ValueError, match="gold, top1, and top5 lengths"):
        classification_report(["A"], ["A", "A"], [["A"]], ["A"])
    with pytest.raises(ValueError, match="gold, top1, and top5 lengths"):
        classification_report(["A"], ["A"], [["A"], ["A"]], ["A"])


def test_metric_zip_helpers_reject_mismatched_cardinality() -> None:
    mismatch = r"zip\(\) argument 2 is shorter than argument 1"
    with pytest.raises(ValueError, match=mismatch):
        _accuracy_counts(["A", "B"], ["A"], [["A"], ["B"]])
    with pytest.raises(ValueError, match=mismatch):
        _true_positives(["A", "B"], ["A"])
    with pytest.raises(ValueError, match=mismatch):
        _accuracy_counts(["A", "B"], ["A", "B"], [["A"]])


def test_metrics_validation_errors_are_exact() -> None:
    from georeset_text_label_benchmark.pilot import metrics

    _assert_value_error(
        lambda: metrics._validate_lengths([], [], []),
        "gold, top1, and top5 lengths must agree and be non-empty",
    )
    _assert_value_error(
        lambda: metrics._validate_candidate_codes([]),
        "candidate codes must be non-empty and unique",
    )
    _assert_value_error(
        lambda: metrics._validate_top5("A", [], {"A"}),
        "top5 must contain one to five unique candidate codes",
    )
    _assert_value_error(
        lambda: metrics._validate_top5("A", ["B"], {"A", "B"}),
        "top1 prediction must be first in top5",
    )
    _assert_value_error(
        lambda: group_breakdown([], ["A"], ["A"], [["A"]]),
        "groups, gold, top1, and top5 lengths must agree",
    )


def test_macro_f1_keeps_positive_scores_when_support_equals_predictions() -> None:
    report = classification_report(["A", "B"], ["A", "B"], [["A"], ["B"]], ["A", "B"])

    assert report["macro_f1_all_candidates"] == pytest.approx(1.0)


def test_metrics_reject_empty_duplicate_and_overlong_candidate_predictions() -> None:
    with pytest.raises(ValueError, match="candidate codes must be non-empty and unique"):
        classification_report(["A"], ["A"], [["A"]], [])
    with pytest.raises(ValueError, match="candidate codes must be non-empty and unique"):
        classification_report(["A"], ["A"], [["A"]], ["A", "A"])
    with pytest.raises(ValueError, match="top5 must contain"):
        classification_report(["A"], ["A"], [[]], ["A"])
    with pytest.raises(ValueError, match="top5 must contain"):
        classification_report(["A"], ["A"], [["A", "A"]], ["A", "B"])
    codes = ["A", "B", "C", "D", "E", "F"]
    with pytest.raises(ValueError, match="top5 must contain"):
        classification_report(["A"], ["A"], [codes], codes)
    with pytest.raises(ValueError, match="groups, gold, top1, and top5 lengths must agree"):
        group_breakdown([], ["A"], ["A"], [["A"]])


def test_group_breakdown_reports_language_counts_and_missing_values() -> None:
    result = group_breakdown(
        ["eng", None, "eng"],
        ["A", "B", "B"],
        ["A", "A", "B"],
        [["A", "B"], ["A", "B"], ["B", "A"]],
    )

    assert result == [
        {"group": "eng", "sample_count": 2, "top1_accuracy": 1.0, "top5_accuracy": 1.0},
        {"group": "missing", "sample_count": 1, "top1_accuracy": 0.0, "top5_accuracy": 1.0},
    ]


def test_average_pool_masks_padding_and_handles_empty_attention_rows() -> None:
    hidden = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [90.0, 90.0]], [[3.0, 3.0], [9.0, 9.0], [8.0, 8.0]]]
    )
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])

    pooled = average_pool(hidden, mask)

    assert torch.equal(pooled[0], torch.tensor([1.0, 1.0]))
    assert torch.equal(pooled[1], torch.tensor([0.0, 0.0]))
    with pytest.raises(ValueError, match="hidden states and attention mask shapes must agree"):
        average_pool(hidden, torch.ones((2, 2)))


def test_average_pool_reports_exact_shape_error() -> None:
    with pytest.raises(
        ValueError, match=re.escape("hidden states and attention mask shapes must agree")
    ) as error:
        average_pool(torch.zeros((1, 2, 3)), torch.ones((1, 1)))
    assert str(error.value) == "hidden states and attention mask shapes must agree"


def test_average_pool_divides_each_sequence_by_its_own_token_count() -> None:
    hidden = torch.tensor([[[2.0, 4.0], [4.0, 2.0]], [[9.0, 3.0], [7.0, 7.0]]])
    mask = torch.tensor([[1, 1], [1, 0]])

    pooled = average_pool(hidden, mask)

    assert torch.equal(pooled, torch.tensor([[3.0, 3.0], [9.0, 3.0]]))


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


def test_encoder_applies_prefix_batches_and_l2_normalization() -> None:
    tokenizer = _RecordingTokenizer()
    model = _RecordingModel()

    result = encode_texts(
        ["one", "two", "three"], tokenizer, model, prefix="query", batch_size=2, max_length=77
    )

    assert tokenizer.calls == [["query: one", "query: two"], ["query: three"]]
    assert tokenizer.options[0] == {
        "max_length": 77,
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    assert model.eval_called
    assert result.shape == (3, 2)
    assert torch.allclose(result, torch.tensor([[1.0, 0.0]]).repeat(3, 1))


def test_encoder_keeps_distinct_text_vectors_in_order_across_batches() -> None:
    texts = ["maple forest", "wet meadow", "lava field", "salt marsh"]
    prefixed = [f"query: {text}" for text in texts]
    identities = {text: index for index, text in enumerate(prefixed, start=1)}

    class DistinctTokenizer:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def __call__(self, batch_texts: list[str], **_kwargs: Any) -> dict[str, torch.Tensor]:
            self.calls.append(batch_texts)
            input_ids = torch.tensor([[identities[text]] for text in batch_texts])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

    class DistinctModel:
        def eval(self) -> DistinctModel:
            return self

        def __call__(self, **batch: torch.Tensor) -> SimpleNamespace:
            basis = torch.tensor(
                [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0], [1.0, 1.0, 0.0]]
            )
            hidden = basis[batch["input_ids"][:, 0] - 1].unsqueeze(1)
            return SimpleNamespace(last_hidden_state=hidden)

    tokenizer = DistinctTokenizer()
    result = encode_texts(
        texts, tokenizer, DistinctModel(), prefix="query", batch_size=3, max_length=77
    )

    assert tokenizer.calls == [prefixed[:3], prefixed[3:]]
    assert torch.allclose(
        result,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2**-0.5, 2**-0.5, 0.0]]),
    )


def test_encoder_uses_default_batch_size_and_euclidean_normalization() -> None:
    tokenizer = _RecordingTokenizer()

    class NonUnitModel(_RecordingModel):
        def __call__(self, **batch: torch.Tensor) -> SimpleNamespace:
            count, token_count = batch["input_ids"].shape
            assert token_count == 2
            hidden = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]]).repeat(count, 1, 1)
            return SimpleNamespace(last_hidden_state=hidden)

    result = encode_texts(
        [f"text {index}" for index in range(20)], tokenizer, NonUnitModel(), prefix="query"
    )

    assert [len(batch) for batch in tokenizer.calls] == [16, 4]
    assert torch.allclose(result, torch.tensor([[0.6, 0.8]]).repeat(20, 1))


def test_encoder_accepts_unit_batch_and_default_max_length() -> None:
    tokenizer = _RecordingTokenizer()
    encode_texts(["one"], tokenizer, _RecordingModel(), prefix="query", batch_size=1)

    assert tokenizer.options == [
        {
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
    ]


def test_embedding_validation_errors_are_exact() -> None:
    from georeset_text_label_benchmark.pilot import embeddings

    _assert_value_error(
        lambda: embeddings._validate_text_values([""]), "texts must be non-empty strings"
    )
    _assert_value_error(
        lambda: embeddings._validate_prefix("other"), "prefix must be query or passage"
    )
    _assert_value_error(
        lambda: embeddings._validate_dimensions(0, 1),
        "batch size and max length must be positive",
    )
    _assert_value_error(
        lambda: embeddings._validate_embedding_shapes(torch.tensor([1]), torch.tensor([1]), []),
        "query and candidate embeddings must be two-dimensional",
    )
    _assert_value_error(
        lambda: embeddings._validate_embedding_shapes(
            torch.zeros((1, 2)), torch.zeros((1, 3)), ["A"]
        ),
        "query and candidate embedding dimensions must agree",
    )
    _assert_value_error(
        lambda: embeddings._validate_embedding_shapes(
            torch.zeros((1, 2)), torch.zeros((1, 2)), ["A", "B"]
        ),
        "candidate codes and candidate embeddings must align",
    )
    _assert_value_error(
        lambda: embeddings._validate_codes([], 0), "candidate codes must be non-empty and unique"
    )
    _assert_value_error(
        lambda: embeddings._validate_codes(["A"], 0), "top_k must fit the candidate set"
    )
    embeddings._validate_dimensions(1, 1)
    embeddings._validate_codes(["A"], 1)


def test_encoder_rejects_empty_text_invalid_prefix_and_batch_size() -> None:
    with pytest.raises(ValueError, match="texts must be non-empty"):
        encode_texts([], _RecordingTokenizer(), _RecordingModel(), prefix="query")
    with pytest.raises(ValueError, match="prefix must be query or passage"):
        encode_texts(["one"], _RecordingTokenizer(), _RecordingModel(), prefix="other")
    with pytest.raises(ValueError, match="batch size and max length must be positive"):
        encode_texts(
            ["one"], _RecordingTokenizer(), _RecordingModel(), prefix="query", batch_size=0
        )
    with pytest.raises(ValueError, match="texts must be non-empty strings"):
        encode_texts([""], _RecordingTokenizer(), _RecordingModel(), prefix="query")
    with pytest.raises(ValueError, match="batch size and max length must be positive"):
        encode_texts(
            ["one"], _RecordingTokenizer(), _RecordingModel(), prefix="query", max_length=0
        )


def test_rank_candidates_uses_code_order_to_break_score_ties() -> None:
    queries = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    result = rank_candidates(queries, candidates, ["B", "A", "C"], top_k=3)

    assert [code for code, _ in result[0]] == ["A", "B", "C"]
    assert result[0][0][1] == pytest.approx(1.0)
    assert result[0][2][1] == pytest.approx(0.0)


def test_rank_candidates_defaults_to_five_results() -> None:
    vectors = torch.eye(7)
    result = rank_candidates(vectors[:1], vectors, [f"C{index}" for index in range(7)])

    assert [code for code, _ in result[0]] == ["C0", "C1", "C2", "C3", "C4"]


def test_rank_candidates_uses_scores_before_lexical_ties() -> None:
    queries = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])

    result = rank_candidates(queries, candidates, ["A", "Z", "M"], top_k=3)

    assert [code for code, _ in result[0]] == ["Z", "M", "A"]
    assert [
        code for code, _ in rank_candidates(queries, candidates, ["A", "Z", "M"], top_k=1)[0]
    ] == ["Z"]


def test_rank_candidates_rejects_bad_shapes_codes_and_top_k() -> None:
    vectors = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="two-dimensional"):
        rank_candidates(torch.tensor([1.0, 0.0]), vectors, ["A"])
    with pytest.raises(ValueError, match="two-dimensional"):
        rank_candidates(vectors, torch.tensor([1.0, 0.0]), ["A"])
    with pytest.raises(ValueError, match="embedding dimensions must agree"):
        rank_candidates(vectors, torch.tensor([[1.0, 0.0, 0.0]]), ["A"])
    with pytest.raises(ValueError, match="candidate codes and candidate embeddings must align"):
        rank_candidates(vectors, vectors, ["A", "B"])
    with pytest.raises(ValueError, match="candidate codes must be non-empty and unique"):
        rank_candidates(vectors, torch.empty((0, 2)), [])
    with pytest.raises(ValueError, match="candidate codes must be non-empty and unique"):
        rank_candidates(vectors, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ["A", "A"])
    with pytest.raises(ValueError, match="top_k must fit the candidate set"):
        rank_candidates(vectors, vectors, ["A"], top_k=0)
    with pytest.raises(ValueError, match="top_k must fit the candidate set"):
        rank_candidates(vectors, vectors, ["A"], top_k=2)


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


def test_runner_executes_frozen_pilot_and_writes_metrics_and_hash_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = _candidate_file()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    selected = _frozen_sample(run_dir, candidates)
    model_dir = _write_model_stubs(tmp_path)
    model_cache = tmp_path / "model-cache"
    tokenizer = _RecordingTokenizer()
    model = _RecordingModel()
    sha256_file_spy = Mock(wraps=runner.sha256_file)

    def download(cache: Path) -> Path:
        assert cache == model_cache
        return model_dir

    def load(path: Path) -> tuple[_RecordingTokenizer, _RecordingModel]:
        assert path == model_dir
        return tokenizer, model

    timer = iter(range(5, 25))
    monkeypatch.setattr(runner, "_download_model", download)
    monkeypatch.setattr(runner, "_load_model", load)
    monkeypatch.setattr(runner, "perf_counter", lambda: next(timer))
    monkeypatch.setattr(runner, "sha256_file", sha256_file_spy)

    result = runner.run_pilot(
        run_dir,
        model_cache,
        computation_commit="a" * 40,
        validation_commit="b" * 40,
        batch_size=2,
        max_length=64,
    )

    table = pq.read_table(run_dir / "predictions.parquet")
    _assert_run_file_names(run_dir)
    _assert_manifest_hash_path_names(sha256_file_spy.call_args_list)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with candidates.open(encoding="utf-8", newline="") as stream:
        candidate_rows = list(csv.DictReader(stream))
    candidate_codes = sorted(row["eunis_code"] for row in candidate_rows)
    names = {row["eunis_code"]: row["eunis_name"] for row in candidate_rows}
    top5_codes = candidate_codes[:5]
    expected_predictions = [
        {
            **row,
            "gold_eunis_code": row["eunis_code"],
            "gold_eunis_name": row["eunis_name"],
            "top1_eunis_code": candidate_codes[0],
            "top1_eunis_name": names[candidate_codes[0]],
            "top1_cosine": 1.0,
            "top5_eunis_codes": top5_codes,
            "top5_eunis_names": [names[code] for code in top5_codes],
            "top5_cosine_scores": [1.0] * 5,
            "correct_top1": row["eunis_code"] == candidate_codes[0],
            "correct_top5": row["eunis_code"] in top5_codes,
        }
        for row in selected
    ]
    assert table.column_names == [
        *sorted(selected[0]),
        "gold_eunis_code",
        "gold_eunis_name",
        "top1_eunis_code",
        "top1_eunis_name",
        "top1_cosine",
        "top5_eunis_codes",
        "top5_eunis_names",
        "top5_cosine_scores",
        "correct_top1",
        "correct_top5",
    ]
    assert table.to_pylist() == expected_predictions
    query_texts = [f"query: {row['sentence']}" for row in selected]
    passage_texts = [f"passage: {row['candidate_text']}" for row in candidate_rows]
    assert tokenizer.calls[:2] == [query_texts[:2], query_texts[2:]]
    assert tokenizer.calls[2:] == [
        passage_texts[index : index + 2] for index in range(0, len(passage_texts), 2)
    ]
    assert len(tokenizer.options) == 81
    assert all(
        options
        == {
            "max_length": 64,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        for options in tokenizer.options
    )
    assert model.eval_called

    gold = [row["eunis_code"] for row in selected]
    top1 = [candidate_codes[0]] * len(selected)
    top5 = [top5_codes] * len(selected)
    correct = sum(actual == predicted for actual, predicted in zip(gold, top1, strict=True))
    top5_hits = sum(actual in predicted for actual, predicted in zip(gold, top5, strict=True))
    class_rows = []
    f1_values = []
    for code in candidate_codes:
        support = sum(actual == code for actual in gold)
        predicted = sum(value == code for value in top1)
        true_positive = sum(
            actual == code and value == code for actual, value in zip(gold, top1, strict=True)
        )
        hits = sum(
            actual == code and code in ranked for actual, ranked in zip(gold, top5, strict=True)
        )
        f1_values.append(2 * true_positive / (support + predicted) if support + predicted else 0.0)
        class_rows.append(
            {
                "eunis_code": code,
                "support": support,
                "top1_correct": true_positive,
                "top1_accuracy": true_positive / support if support else None,
                "top5_hits": hits,
                "top5_recall": hits / support if support else None,
                "eunis_name": names[code],
            }
        )
    expected_overall = {
        "sample_count": 3,
        "candidate_class_count": 158,
        "top1_accuracy": correct / 3,
        "top5_accuracy": top5_hits / 3,
        "macro_f1_all_candidates": sum(f1_values) / len(candidate_codes),
        "macro_f1_definition": (
            "Unweighted mean of per-code F1 over all candidate codes; codes with no gold "
            "support and no predictions contribute zero."
        ),
    }
    expected_metrics = {
        "overall": expected_overall,
        "by_eunis_class": class_rows,
        "by_language": [
            {
                "group": "eng",
                "sample_count": 3,
                "top1_accuracy": correct / 3,
                "top5_accuracy": top5_hits / 3,
            }
        ],
        "top_confusions": (
            []
            if candidate_codes[0] == "T11"
            else [
                {"gold_eunis_code": "T11", "predicted_eunis_code": candidate_codes[0], "count": 3}
            ]
        ),
        "scope_note": (
            "All rows are sampled from the existing positive Description/EUNIS overlap. "
            "The EUNIS assignment is polygon-level context, not sentence-level truth; "
            "its scientific validation remains unconfirmed."
        ),
    }
    assert metrics == expected_metrics
    assert (run_dir / "metrics.json").read_bytes() == (
        json.dumps(expected_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    parquet_metadata = pq.ParquetFile(run_dir / "predictions.parquet").metadata
    assert all(
        parquet_metadata.row_group(0).column(index).compression == "ZSTD"
        for index in range(parquet_metadata.num_columns)
    )

    with (run_dir / "frozen_sample.json").open(encoding="utf-8") as stream:
        frozen = json.load(stream)
    expected_inventory = [
        {"file": name, "bytes": len(name), "sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in MODEL_FILES
    ]
    with candidates.open(encoding="utf-8", newline="") as stream:
        taxonomy = list(csv.DictReader(stream))
    expected_manifest = {
        "format_version": 1,
        "computation_commit": "a" * 40,
        "validation_commit": "b" * 40,
        "source": frozen["source"],
        "source_coverage": frozen["source_coverage"],
        "sample": frozen["selection"],
        "candidate_labels": {
            **frozen["candidate_labels"],
            "classification_releases": sorted({row["classification_release"] for row in taxonomy}),
            "classification_sources": sorted({row["classification_source"] for row in taxonomy}),
            "taxonomy_archive_sha256": sorted({row["source_sha256"] for row in taxonomy}),
            "license": "CC-BY-4.0, European Environment Agency",
        },
        "model": {
            "repository": "intfloat/multilingual-e5-small",
            "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
            "license": "MIT",
            "files": expected_inventory,
            "method": "query/passage-prefixed mean pooling with attention mask and L2 normalization",
        },
        "settings": {
            "query_prefix": "query: ",
            "candidate_prefix": "passage: ",
            "maximum_tokens": 64,
            "batch_size": 2,
            "candidate_count": 158,
            "candidate_text": "EUNIS English name + newline + authoritative EEA description",
            "scoring": "cosine similarity of L2-normalized embeddings; top-5 ties sorted by code",
            "training_or_finetuning": False,
            "gold_label_used_in_input": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "pyarrow": version("pyarrow"),
            "inference_device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "timings_seconds": {
            "model_download": 1,
            "model_load_setup": 1,
            "sentence_encoding_cpu": 1,
            "candidate_encoding_cpu": 1,
            "similarity_ranking_cpu": 1,
            "total_cpu_inference": 7,
            "cpu_inference_excludes_model_download_and_load": True,
        },
        "outputs_sha256": {
            "frozen_sample.json": _file_sha256(run_dir / "frozen_sample.json"),
            "candidate_labels.csv": _file_sha256(run_dir / "candidate_labels.csv"),
            "predictions.parquet": _file_sha256(run_dir / "predictions.parquet"),
            "metrics.json": _file_sha256(run_dir / "metrics.json"),
        },
        "limitations": [
            "A 100-sentence pilot is not a comprehensive evaluation.",
            "EUNIS labels are polygon-level and may not describe the exact sentence evidence.",
            "Final scientific validation of the EUNIS reference assignments is unconfirmed.",
            "Candidate names and descriptions are English; source sentences span languages.",
            "Results are zero-shot label ranking, not model training or fine-tuning.",
        ],
    }
    assert manifest == expected_manifest
    assert result == {
        "run_dir": str(run_dir),
        "sample_count": 3,
        "candidate_count": 158,
        "metrics": expected_overall,
        "cpu_inference_seconds": 7,
        "model_download_seconds": 1,
        "model_load_seconds": 1,
        "predictions_sha256": _file_sha256(run_dir / "predictions.parquet"),
        "manifest_sha256": _file_sha256(run_dir / "manifest.json"),
    }


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


@pytest.mark.parametrize("failing_name", runner.OUTPUT_FILES)
def test_runner_does_not_leave_partial_outputs_when_publication_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_name: str
) -> None:
    candidates = _candidate_file()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _frozen_sample(run_dir, candidates)
    frozen_before = {
        name: (run_dir / name).read_bytes()
        for name in ("candidate_labels.csv", "frozen_sample.json")
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in MODEL_FILES:
        (model_dir / name).write_bytes(name.encode("utf-8"))
    monkeypatch.setattr(runner, "_download_model", lambda _cache: model_dir)
    monkeypatch.setattr(
        runner, "_load_model", lambda _path: (_RecordingTokenizer(), _RecordingModel())
    )
    monkeypatch.setattr(runner, "perf_counter", iter(range(5, 25)).__next__)
    _inject_e5_output_failure(monkeypatch, failing_name)

    with pytest.raises(OSError, match="injected publication failure"):
        runner.run_pilot(
            run_dir,
            tmp_path / "model-cache",
            computation_commit="a" * 40,
            validation_commit="b" * 40,
            batch_size=2,
            max_length=64,
        )

    assert not any((run_dir / name).exists() for name in runner.OUTPUT_FILES)
    assert {name: (run_dir / name).read_bytes() for name in frozen_before} == frozen_before
    staging_dirs = list(run_dir.glob(".pilot-output-*"))
    assert len(staging_dirs) == 1
    assert (staging_dirs[0] / "manifest.json").exists() is (failing_name == "manifest.json")


def test_runner_resumes_a_completed_staged_publication_without_reloading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from georeset_text_label_benchmark.pilot import publication

    candidates = _candidate_file()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _frozen_sample(run_dir, candidates)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in MODEL_FILES:
        (model_dir / name).write_bytes(name.encode("utf-8"))
    monkeypatch.setattr(runner, "_download_model", lambda _cache: model_dir)
    monkeypatch.setattr(
        runner, "_load_model", lambda _path: (_RecordingTokenizer(), _RecordingModel())
    )
    monkeypatch.setattr(runner, "perf_counter", iter(range(5, 25)).__next__)
    original_link = publication.os.link
    failed_once = False

    def fail_metrics_once(source: Path, target: Path) -> None:
        nonlocal failed_once
        if target.parent == run_dir and target.name == "metrics.json" and not failed_once:
            failed_once = True
            raise OSError("transient publication failure")
        original_link(source, target)

    monkeypatch.setattr(publication.os, "link", fail_metrics_once)

    with pytest.raises(OSError, match="transient publication failure"):
        runner.run_pilot(
            run_dir,
            tmp_path / "model-cache",
            computation_commit="a" * 40,
            validation_commit="b" * 40,
            batch_size=2,
            max_length=64,
        )

    staging_dirs = list(run_dir.glob(".pilot-output-*"))
    assert len(staging_dirs) == 1
    staging = staging_dirs[0]
    assert all((staging / name).is_file() for name in runner.OUTPUT_FILES)
    assert (run_dir / "predictions.parquet").stat().st_ino == (
        staging / "predictions.parquet"
    ).stat().st_ino
    assert not (run_dir / "manifest.json").exists()

    monkeypatch.setattr(
        runner,
        "_download_model",
        lambda _cache: pytest.fail("a valid staged prediction run must resume before model load"),
    )
    monkeypatch.setattr(
        runner,
        "_load_model",
        lambda _path: pytest.fail("a valid staged prediction run must resume before model load"),
    )

    result = runner.run_pilot(
        run_dir,
        tmp_path / "model-cache",
        computation_commit="a" * 40,
        validation_commit="b" * 40,
        batch_size=2,
        max_length=64,
    )

    published_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    published_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    timings = published_manifest["timings_seconds"]
    assert result == {
        "run_dir": str(run_dir),
        "sample_count": 3,
        "candidate_count": 158,
        "metrics": published_metrics["overall"],
        "cpu_inference_seconds": timings["total_cpu_inference"],
        "model_download_seconds": timings["model_download"],
        "model_load_seconds": timings["model_load_setup"],
        "predictions_sha256": runner.sha256_file(run_dir / "predictions.parquet"),
        "manifest_sha256": runner.sha256_file(run_dir / "manifest.json"),
    }
    assert (run_dir / "manifest.json").is_file()
    assert result["manifest_sha256"] == runner.sha256_file(run_dir / "manifest.json")
    with pytest.raises(FileExistsError, match="outputs already exist"):
        runner.run_pilot(
            run_dir,
            tmp_path / "model-cache",
            computation_commit="a" * 40,
            validation_commit="b" * 40,
            batch_size=2,
            max_length=64,
        )


def test_runner_rejects_a_staged_protocol_mismatch_with_its_path(tmp_path: Path) -> None:
    staging = tmp_path / ".pilot-output-staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match this frozen run and settings") as error:
        runner._resume_staged_run(
            tmp_path,
            staging,
            {"source": {}, "selection": {}, "candidate_labels": {}},
            [],
            "a" * 40,
            "b" * 40,
            1,
            1,
        )

    assert str(error.value) == (
        f"retained E5 staging manifest does not match this frozen run and settings: {staging}"
    )


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_resume_staged_run_names_the_e5_adapter_in_manifest_errors(
    tmp_path: Path, contents: str
) -> None:
    staging = tmp_path / ".pilot-output-staging"
    staging.mkdir()
    (staging / "manifest.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="retained E5 staging") as error:
        runner._resume_staged_run(
            tmp_path,
            staging,
            {"source": {}, "selection": {}, "candidate_labels": {}},
            [],
            "a" * 40,
            "b" * 40,
            1,
            1,
        )

    expected = (
        f"retained E5 staging directory is incomplete or unreadable: {staging}"
        if contents == "{"
        else f"retained E5 staging manifest must be a JSON object: {staging}"
    )
    assert str(error.value) == expected


def test_runner_staged_manifest_reader_uses_explicit_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    manifest_path = staging / "manifest.json"
    manifest_path.write_text('{"name": "Forêt"}', encoding="utf-8")
    observed: list[str | None] = []
    original_read_text = Path.read_text

    def record_encoding(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == manifest_path:
            observed.append(kwargs.get("encoding"))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_encoding)

    assert runner._read_staged_manifest(staging, "E5") == {"name": "Forêt"}
    assert observed == ["utf-8"]


def test_runner_staged_manifest_reader_reports_exact_invalid_json_error(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or unreadable") as error:
        runner._read_staged_manifest(staging, "E5")

    assert str(error.value) == (
        f"retained E5 staging directory is incomplete or unreadable: {staging}"
    )


def test_runner_staged_manifest_reader_rejects_non_object_json(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object") as error:
        runner._read_staged_manifest(staging, "E5")

    assert str(error.value) == f"retained E5 staging manifest must be a JSON object: {staging}"


def test_runner_staged_summary_returns_the_complete_result_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    staging = tmp_path / "staging"
    run_dir.mkdir()
    staging.mkdir()
    (staging / "metrics.json").write_text('{"overall": {"accuracy": 0.5}}', encoding="utf-8")
    (run_dir / "predictions.parquet").write_bytes(b"predictions")
    (run_dir / "manifest.json").write_bytes(b"manifest")
    sample = {"selection": {"sample_size": 2}}
    candidates = [{"eunis_code": "T11"}, {"eunis_code": "T12"}]
    manifest = {
        "timings_seconds": {
            "total_cpu_inference": 3.5,
            "model_download": 1.25,
            "model_load_setup": 2.0,
        }
    }

    result = runner._staged_run_summary(run_dir, staging, sample, candidates, manifest)

    assert result == {
        "run_dir": str(run_dir),
        "sample_count": 2,
        "candidate_count": 2,
        "metrics": {"accuracy": 0.5},
        "cpu_inference_seconds": 3.5,
        "model_download_seconds": 1.25,
        "model_load_seconds": 2.0,
        "predictions_sha256": runner.sha256_file(run_dir / "predictions.parquet"),
        "manifest_sha256": runner.sha256_file(run_dir / "manifest.json"),
    }


def test_staged_e5_summary_reads_metrics_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    staging = tmp_path / "staging"
    run_dir.mkdir()
    staging.mkdir()
    metrics_path = staging / "metrics.json"
    metrics_path.write_text('{"overall": {"name": "Forêt"}}', encoding="utf-8")
    (run_dir / "predictions.parquet").write_bytes(b"predictions")
    (run_dir / "manifest.json").write_bytes(b"manifest")
    sample = {"selection": {"sample_size": 1}}
    manifest = {
        "timings_seconds": {
            "total_cpu_inference": 1.0,
            "model_download": 2.0,
            "model_load_setup": 3.0,
        }
    }
    original_read_text = Path.read_text
    observed_encodings: list[str | None] = []

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == metrics_path:
            observed_encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    result = runner._staged_run_summary(run_dir, staging, sample, [], manifest)

    assert result["metrics"] == {"name": "Forêt"}
    assert observed_encodings == ["utf-8"]


def test_staged_e5_protocol_validates_identity_and_settings_exactly(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    sample = {
        "source": {"dataset": "overlap"},
        "selection": {"seed": 42},
        "candidate_labels": {"file": "candidate_labels.csv"},
    }
    candidates = [
        {
            "classification_release": "release",
            "classification_source": "source",
            "source_sha256": "hash",
        }
    ]
    manifest = {
        "computation_commit": "a" * 40,
        "validation_commit": "b" * 40,
        "source": sample["source"],
        "sample": sample["selection"],
        "candidate_labels": runner._candidate_provenance(sample, candidates),
        "settings": {"batch_size": 1, "maximum_tokens": 64},
    }

    runner._validate_staged_run_protocol(
        staging, manifest, sample, candidates, "a" * 40, "b" * 40, 1, 64
    )

    for invalid_manifest in (
        {**manifest, "settings": {"batch_size": 2, "maximum_tokens": 64}},
        {**manifest, "computation_commit": "c" * 40},
        {**manifest, "settings": []},
    ):
        with pytest.raises(
            ValueError, match="does not match this frozen run and settings"
        ) as error:
            runner._validate_staged_run_protocol(
                staging, invalid_manifest, sample, candidates, "a" * 40, "b" * 40, 1, 64
            )
        assert str(error.value) == (
            f"retained E5 staging manifest does not match this frozen run and settings: {staging}"
        )


@pytest.mark.parametrize(("batch_size", "max_length"), [(0, 1), (1, 0)])
def test_run_option_validator_checks_each_positive_boundary(
    batch_size: int, max_length: int
) -> None:
    with pytest.raises(ValueError, match="batch size and max length must be positive") as error:
        runner._validate_run_options("a" * 40, "b" * 40, batch_size, max_length)

    assert str(error.value) == "batch size and max length must be positive"


def test_run_option_validator_accepts_one_as_a_positive_boundary() -> None:
    runner._validate_run_options("a" * 40, "b" * 40, 1, 1)


def test_run_option_validator_reports_exact_commit_field_names() -> None:
    with pytest.raises(ValueError, match="computation_commit") as computation_error:
        runner._validate_run_options("A" * 40, "b" * 40, 1, 1)
    assert str(computation_error.value) == (
        "computation_commit must be a 40-character lowercase Git commit SHA"
    )

    with pytest.raises(ValueError, match="validation_commit") as validation_error:
        runner._validate_run_options("a" * 40, "B" * 40, 1, 1)
    assert str(validation_error.value) == (
        "validation_commit must be a 40-character lowercase Git commit SHA"
    )


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


def test_staged_e5_hash_validator_binds_frozen_inputs_and_staged_outputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    staging = tmp_path / "staging"
    run_dir.mkdir()
    staging.mkdir()
    frozen_sample = run_dir / "frozen_sample.json"
    candidate_labels = run_dir / "candidate_labels.csv"
    predictions = staging / "predictions.parquet"
    metrics = staging / "metrics.json"
    for path, content in (
        (frozen_sample, b"frozen sample"),
        (candidate_labels, b"candidate labels"),
        (predictions, b"predictions"),
        (metrics, b"metrics"),
    ):
        path.write_bytes(content)
    hashes = {
        path.name: runner.sha256_file(path)
        for path in (frozen_sample, candidate_labels, predictions, metrics)
    }
    manifest = {"outputs_sha256": hashes}

    with pytest.raises(ValueError, match="retained E5 staging input hash mismatch") as error:
        runner._validate_staged_run_hashes(run_dir, staging, {"outputs_sha256": {}})
    assert str(error.value) == (
        f"retained E5 staging input hash mismatch for frozen_sample.json: {staging}"
    )

    runner._validate_staged_run_hashes(run_dir, staging, manifest)


def test_staged_hash_validation_requires_a_hash_mapping(tmp_path: Path) -> None:
    staging = tmp_path / "staging"

    with pytest.raises(ValueError, match="has no output hashes") as error:
        runner._validate_staged_hashes(staging, {}, {}, {}, "E5")

    assert str(error.value) == f"retained E5 staging manifest has no output hashes: {staging}"


def test_no_existing_outputs_error_is_exact(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").touch()

    with pytest.raises(FileExistsError) as error:
        runner._validate_no_existing_outputs(tmp_path)

    assert str(error.value) == "pilot prediction outputs already exist"


def test_runner_checks_output_mount_before_downloading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from georeset_text_label_benchmark.pilot import publication

    candidates = _candidate_file()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _frozen_sample(run_dir, candidates)

    def unsupported_link(_source: Path, _target: Path) -> None:
        raise OSError(22, "this filesystem rejects exclusive hard links")

    monkeypatch.setattr(publication.os, "link", unsupported_link)
    monkeypatch.setattr(
        runner,
        "_download_model",
        lambda _cache: pytest.fail("unsupported publication must be rejected before download"),
    )

    with pytest.raises(OSError, match="output filesystem must support exclusive hard links"):
        runner.run_pilot(
            run_dir,
            tmp_path / "model-cache",
            computation_commit="a" * 40,
            validation_commit="b" * 40,
            batch_size=2,
            max_length=64,
        )

    assert not any((run_dir / name).exists() for name in runner.OUTPUT_FILES)


def test_runner_preserves_a_file_created_after_output_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from georeset_text_label_benchmark.pilot import publication

    candidates = _candidate_file()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _frozen_sample(run_dir, candidates)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in MODEL_FILES:
        (model_dir / name).write_bytes(name.encode("utf-8"))
    monkeypatch.setattr(runner, "_download_model", lambda _cache: model_dir)
    monkeypatch.setattr(
        runner, "_load_model", lambda _path: (_RecordingTokenizer(), _RecordingModel())
    )
    monkeypatch.setattr(runner, "perf_counter", iter(range(5, 25)).__next__)
    publish = publication.publish_files

    def create_racing_file(staging: Path, destination: Path, names: Any) -> None:
        (destination / "metrics.json").write_bytes(b"competing writer")
        publish(staging, destination, names)

    monkeypatch.setattr(runner, "publish_files", create_racing_file)

    with pytest.raises(FileExistsError):
        runner.run_pilot(
            run_dir,
            tmp_path / "model-cache",
            computation_commit="a" * 40,
            validation_commit="b" * 40,
            batch_size=2,
            max_length=64,
        )

    assert (run_dir / "metrics.json").read_bytes() == b"competing writer"
    assert (run_dir / "predictions.parquet").is_file()
    assert not (run_dir / "manifest.json").exists()
    assert len(list(run_dir.glob(".pilot-output-*"))) == 1

    with pytest.raises(FileExistsError):
        runner.run_pilot(
            run_dir,
            tmp_path / "model-cache",
            computation_commit="a" * 40,
            validation_commit="b" * 40,
            batch_size=2,
            max_length=64,
        )
    assert (run_dir / "metrics.json").read_bytes() == b"competing writer"


def test_runner_keeps_distinct_query_vectors_attached_to_frozen_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = _candidate_file()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    selected = _frozen_sample(run_dir, candidates)
    with candidates.open(encoding="utf-8", newline="") as stream:
        candidate_rows = list(csv.DictReader(stream))

    expected_codes, token_ids, vectors_by_id = _identity_embedding_vectors(selected, candidate_rows)

    class IdentityTokenizer:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def __call__(self, texts: list[str], **_kwargs: Any) -> dict[str, torch.Tensor]:
            self.calls.append(texts)
            input_ids = torch.tensor([[token_ids[text]] for text in texts])
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    class IdentityModel:
        def eval(self) -> IdentityModel:
            return self

        def __call__(self, **batch: torch.Tensor) -> SimpleNamespace:
            vectors = torch.stack([vectors_by_id[int(index)] for index in batch["input_ids"][:, 0]])
            return SimpleNamespace(last_hidden_state=vectors.unsqueeze(1))

    model_dir = _write_model_stubs(tmp_path)
    tokenizer = IdentityTokenizer()
    monkeypatch.setattr(runner, "_download_model", lambda _cache: model_dir)
    monkeypatch.setattr(runner, "_load_model", lambda _path: (tokenizer, IdentityModel()))
    monkeypatch.setattr(runner, "perf_counter", iter(range(100)).__next__)

    runner.run_pilot(
        run_dir,
        tmp_path / "model-cache",
        computation_commit="a" * 40,
        validation_commit="b" * 40,
        batch_size=2,
        max_length=64,
    )

    observed = pq.read_table(run_dir / "predictions.parquet").to_pylist()
    assert [call for call in tokenizer.calls if call[0].startswith("query:")] == [
        [f"query: {row['sentence']}" for row in selected[:2]],
        [f"query: {row['sentence']}" for row in selected[2:]],
    ]
    assert [
        (row["sample_id"], row["sentence"], row["top1_eunis_code"], row["top1_cosine"])
        for row in observed
    ] == _identity_expected_predictions(selected, expected_codes)


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


def _write_model_stubs(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in MODEL_FILES:
        (model_dir / name).write_bytes(name.encode("utf-8"))
    return model_dir


def test_runner_preflight_rejects_bad_commits_settings_and_existing_outputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="computation_commit must be"):
        runner._validate_run_inputs(tmp_path, "bad", "b" * 40, 16, 512)
    with pytest.raises(ValueError, match="validation_commit must be"):
        runner._validate_run_inputs(tmp_path, "a" * 40, "A" * 40, 16, 512)
    with pytest.raises(ValueError, match="batch size and max length"):
        runner._validate_run_inputs(tmp_path, "a" * 40, "b" * 40, 0, 512)
    with pytest.raises(ValueError, match="batch size and max length"):
        runner._validate_run_inputs(tmp_path, "a" * 40, "b" * 40, 16, 0)
    (tmp_path / "metrics.json").touch()
    with pytest.raises(FileExistsError, match="outputs already exist"):
        runner._validate_run_inputs(tmp_path, "a" * 40, "b" * 40, 16, 512)


def test_runner_preflight_accepts_positive_minimum_settings(tmp_path: Path) -> None:
    runner._validate_run_inputs(tmp_path, "a" * 40, "b" * 40, 1, 1)


def test_runner_validation_errors_are_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("candidate", encoding="utf-8")
    source_hash = runner.sha256_file(source)
    rows = select_distinct_sample([_row(1, "one"), _row(2, "two")], size=2, seed=42)
    reversed_rows = list(reversed(rows))
    sample = {"selection": {"sample_size": 2, "seed": 42}}
    sample_with_codes = {"candidate_labels": {"codes": ["T12"]}}
    candidates_dir = tmp_path / "bad-candidates"
    candidates_dir.mkdir()
    (candidates_dir / "candidate_labels.csv").write_text("bad", encoding="utf-8")

    checks = [
        (
            lambda: runner._validate_taxonomy_header(None),
            "candidate CSV schema does not match the pinned pilot schema",
        ),
        (
            lambda: runner._validate_freeze_inputs(source, candidates, source_hash, None, "bad"),
            "source revision is not the pinned overlap snapshot",
        ),
        (
            lambda: runner._validate_freeze_inputs(
                source, candidates, "0" * 64, None, OVERLAP_REVISION
            ),
            "overlap Parquet SHA-256 does not match the pinned input",
        ),
        (
            lambda: runner._validate_freeze_inputs(
                source, candidates, source_hash, "0" * 64, OVERLAP_REVISION
            ),
            "candidate CSV SHA-256 does not match the pinned input",
        ),
        (
            lambda: runner._validate_frozen_source(
                {"dataset": "different", "revision": OVERLAP_REVISION, "sha256": "0" * 64}
            ),
            "frozen sample uses a different overlap dataset revision",
        ),
        (
            lambda: runner._validate_frozen_source(
                {"dataset": OVERLAP_DATASET, "revision": OVERLAP_REVISION, "sha256": "0" * 64}
            ),
            "frozen sample uses a different overlap Parquet hash",
        ),
        (
            lambda: runner._validate_frozen_row_count([], {"sample_size": 1}),
            "frozen sample row count does not match its declaration",
        ),
        (
            lambda: runner._require_unique_frozen_values(["same", "same"]),
            "frozen sample must have unique IDs, sentence hashes, and polygons",
        ),
        (
            lambda: runner._validate_frozen_ids(["sample-id"], {"sample_ids_sha256": "0" * 64}),
            "frozen sample ID checksum mismatch",
        ),
        (
            lambda: runner._validate_frozen_selection(reversed_rows, sample["selection"]),
            "frozen rows do not match the deterministic sampling protocol",
        ),
        (
            lambda: runner._candidate_rows(
                candidates_dir,
                {"candidate_labels": {"file": "candidate_labels.csv", "sha256": "0" * 64}},
            ),
            "candidate label file checksum mismatch",
        ),
        (
            lambda: runner._validate_candidate_code_list(
                [{"eunis_code": "T11"}], sample_with_codes
            ),
            "candidate code list does not match the frozen protocol",
        ),
        (
            lambda: runner._validate_run_inputs(tmp_path, "a" * 40, "b" * 40, 0, 512),
            "batch size and max length must be positive",
        ),
    ]
    for action, expected in checks:
        _assert_value_error(action, expected)

    (tmp_path / "metrics.json").touch()
    with pytest.raises(FileExistsError) as error:
        runner._validate_run_inputs(tmp_path, "a" * 40, "b" * 40, 16, 512)
    assert str(error.value) == "pilot prediction outputs already exist"

    _assert_value_error(
        lambda: runner._validate_commit("a" * 39 + "X", "computation_commit"),
        "computation_commit must be a 40-character lowercase Git commit SHA",
    )


def test_confusion_analysis_counts_errors_only_and_breaks_ties_by_code() -> None:
    result = runner._confusion_pairs(["A", "B", "A"], ["A", "C", "B"])

    assert result == [
        {"gold_eunis_code": "A", "predicted_eunis_code": "B", "count": 1},
        {"gold_eunis_code": "B", "predicted_eunis_code": "C", "count": 1},
    ]


def test_confusion_analysis_prioritizes_frequency_then_gold_and_predicted_code() -> None:
    frequent_first = runner._confusion_pairs(["B", "B", "A"], ["A", "A", "Z"])
    tie_break = runner._confusion_pairs(["A", "B"], ["Z", "A"])

    assert frequent_first == [
        {"gold_eunis_code": "B", "predicted_eunis_code": "A", "count": 2},
        {"gold_eunis_code": "A", "predicted_eunis_code": "Z", "count": 1},
    ]
    assert tie_break == [
        {"gold_eunis_code": "A", "predicted_eunis_code": "Z", "count": 1},
        {"gold_eunis_code": "B", "predicted_eunis_code": "A", "count": 1},
    ]


def test_confusion_analysis_limits_output_to_ten_pairs() -> None:
    codes = [f"C{index:02}" for index in range(11)]

    result = runner._confusion_pairs(codes, [f"P{index:02}" for index in range(11)])

    assert len(result) == 10
    assert [row["gold_eunis_code"] for row in result] == codes[:10]


def test_runner_pair_and_prediction_helpers_reject_mismatched_cardinality() -> None:
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter than argument 1"):
        runner._confusion_pairs(["A", "B"], ["A"])
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter than argument 1"):
        runner._prediction_rows(
            [
                {"eunis_code": "A", "eunis_name": "A"},
                {"eunis_code": "B", "eunis_name": "B"},
            ],
            [{"eunis_code": "A", "eunis_name": "A"}],
            [[("A", 1.0)]],
        )


def test_model_loader_uses_safe_pinned_local_files_and_checks_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    calls: list[dict[str, Any]] = []

    class Model(_RecordingModel):
        config = SimpleNamespace(model_type="bert", hidden_size=384)

        def to(self, device: torch.device) -> Model:
            calls.append({"device": str(device)})
            return self

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(
            lambda path, **kwargs: (
                calls.append({"tokenizer_path": path, "tokenizer": kwargs}) or _RecordingTokenizer()
            )
        ),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        staticmethod(
            lambda path, **kwargs: calls.append({"model_path": path, "model": kwargs}) or Model()
        ),
    )
    tokenizer, model = runner._load_model(tmp_path)

    assert isinstance(tokenizer, _RecordingTokenizer)
    assert model.eval_called
    assert calls == [
        {
            "tokenizer_path": tmp_path,
            "tokenizer": {"local_files_only": True, "trust_remote_code": False},
        },
        {
            "model_path": tmp_path,
            "model": {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
            },
        },
        {"device": "cpu"},
    ]

    class WrongModel(Model):
        config = SimpleNamespace(model_type="bert", hidden_size=768)

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        staticmethod(lambda path, **kwargs: WrongModel()),
    )
    with pytest.raises(ValueError, match="does not match multilingual-e5-small") as error:
        runner._load_model(tmp_path)
    assert str(error.value) == "downloaded checkpoint does not match multilingual-e5-small"

    class WrongTypeModel(Model):
        config = SimpleNamespace(model_type="other", hidden_size=384)

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        staticmethod(lambda path, **kwargs: WrongTypeModel()),
    )
    with pytest.raises(ValueError, match="does not match multilingual-e5-small") as error:
        runner._load_model(tmp_path)
    assert str(error.value) == "downloaded checkpoint does not match multilingual-e5-small"


def test_model_download_is_pinned_and_inventory_requires_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    assert MODEL_FILES == (
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: observed.update(kwargs) or str(tmp_path),
    )
    assert runner._download_model(tmp_path / "cache") == tmp_path
    assert observed == {
        "repo_id": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "cache_dir": str(tmp_path / "cache"),
        "allow_patterns": list(MODEL_FILES),
        "token": False,
    }
    for name in MODEL_FILES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    assert runner._model_inventory(tmp_path) == [
        {"file": name, "bytes": len(name), "sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in MODEL_FILES
    ]
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="required pinned model file"):
        runner._model_inventory(tmp_path)


def test_cli_parser_help_is_stable_and_defaults_are_pinned() -> None:
    parser = cli._parser()
    assert parser.format_help() == (
        "usage: georeset-pilot [-h] {freeze,run,run-dspark,run-dspark-smoke} ...\n\n"
        "positional arguments:\n"
        "  {freeze,run,run-dspark,run-dspark-smoke}\n"
        "    freeze              freeze selected rows before inference\n"
        "    run                 run frozen zero-shot candidate ranking\n"
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
        parser.parse_args(["run", "--validation-commit", "b" * 40])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["run", "--computation-commit", "a" * 40])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["run-dspark"])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            ["run-dspark", "--computation-commit", "a" * 40, "--validation-commit", "b" * 40]
        )

    frozen = parser.parse_args(["freeze", "--source-parquet", "overlap.parquet"])
    assert frozen.source_parquet == Path("overlap.parquet")
    assert frozen.candidate_csv == Path("pilot_data/eunis_candidate_labels.csv")
    assert frozen.output_dir == Path("artifacts/e5-small-100-seed42")
    assert frozen.sample_size == 100
    assert frozen.seed == 42
    running = parser.parse_args(
        ["run", "--computation-commit", "a" * 40, "--validation-commit", "b" * 40]
    )
    assert running.run_dir == Path("artifacts/e5-small-100-seed42")
    assert running.model_cache == Path(".cache/huggingface")
    assert running.batch_size == 16
    assert running.max_length == 512
    dspark_running = parser.parse_args(
        [
            "run-dspark",
            "--smoke-output-dir",
            "smoke-evidence",
            "--computation-commit",
            "a" * 40,
            "--validation-commit",
            "b" * 40,
        ]
    )
    assert dspark_running.run_dir == Path("artifacts/e5-small-100-seed42")
    assert dspark_running.smoke_output_dir == Path("smoke-evidence")
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
    explicit_run = parser.parse_args(
        [
            "run",
            "--run-dir",
            "custom-run",
            "--model-cache",
            "custom-cache",
            "--computation-commit",
            "a" * 40,
            "--validation-commit",
            "b" * 40,
            "--batch-size",
            "3",
            "--max-length",
            "32",
        ]
    )
    assert explicit_run.run_dir == Path("custom-run")
    assert explicit_run.model_cache == Path("custom-cache")
    assert explicit_run.batch_size == 3
    assert explicit_run.max_length == 32


def test_cli_dispatches_exact_freeze_and_run_settings_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def freeze(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(("freeze", args, kwargs))
        return {"z": "é", "a": 1}

    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(("run", args, kwargs))
        return {"z": "β", "a": 2}

    monkeypatch.setattr(cli, "freeze_sample", freeze)
    assert cli.main(["freeze", "--source-parquet", "overlap.parquet"]) == 0
    assert capsys.readouterr().out == '{\n  "a": 1,\n  "z": "é"\n}\n'
    assert calls[-1] == (
        "freeze",
        (
            Path("overlap.parquet"),
            Path("pilot_data/eunis_candidate_labels.csv"),
            Path("artifacts/e5-small-100-seed42"),
        ),
        {
            "expected_source_sha256": OVERLAP_PARQUET_SHA256,
            "source_revision": OVERLAP_REVISION,
            "size": 100,
            "seed": 42,
        },
    )
    monkeypatch.setattr(cli, "run_pilot", run)
    assert cli.main(["run", "--computation-commit", "a" * 40, "--validation-commit", "b" * 40]) == 0
    assert capsys.readouterr().out == '{\n  "a": 2,\n  "z": "β"\n}\n'
    assert calls[-1] == (
        "run",
        (Path("artifacts/e5-small-100-seed42"), Path(".cache/huggingface")),
        {
            "computation_commit": "a" * 40,
            "validation_commit": "b" * 40,
            "batch_size": 16,
            "max_length": 512,
        },
    )
