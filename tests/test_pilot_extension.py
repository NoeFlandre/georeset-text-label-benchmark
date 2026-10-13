"""Tests for extending the published 100-row sample without changing it."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset_text_label_benchmark.output_schema import LABELLED_SCHEMA
from georeset_text_label_benchmark.pilot import dspark_runner, geo_sampling, runner
from georeset_text_label_benchmark.pilot.geo_sampling import (
    extend_geographic_sample,
    h3_cell_of,
    h3_centre_of,
    select_geographic_sample,
)

BASE_SAMPLE = Path(__file__).resolve().parents[1] / "artifacts/pilot-100-seed42/frozen_sample.json"


def _row(index: int, decision: str, lat: float, lon: float) -> dict[str, Any]:
    sentence = f"sentence {index}"
    return {
        "source_pbf": "test-latest",
        "osm_type": "way",
        "osm_id": index,
        "description_identity": f"desc-{index}",
        "tag_key": "description",
        "sentence_index": 0,
        "text_sha256": hashlib.sha256(sentence.encode()).hexdigest(),
        "sentence": sentence,
        "language_code": "eng",
        "eunis_code": "Q11",
        "decision": decision,
        "bbox_min_x": lon - 0.01,
        "bbox_max_x": lon + 0.01,
        "bbox_min_y": lat - 0.01,
        "bbox_max_y": lat + 0.01,
    }


def _grid_rows(size: int = 16) -> list[dict[str, Any]]:
    rows = []
    index = 0
    for decision in ("yes", "no"):
        for i in range(size):
            for j in range(size):
                index += 1
                lat = 40.0 + 1.5 * i + (0.7 if decision == "no" else 0.0)
                lon = -4.0 + 2.0 * j + (0.9 if decision == "no" else 0.0)
                rows.append(_row(index, decision, lat, lon))
    return rows


def _simple_cell_of(lat: float, _lon: float) -> str:
    return str(int(lat))


def _simple_centre_of(cell: str) -> tuple[float, float]:
    return float(cell), 0.0


def _candidate_file() -> Path:
    return Path(__file__).resolve().parents[1] / "pilot_data/eunis_candidate_labels.csv"


def _assert_value_error(message: str, action: Any) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$") as error:
        action()
    assert str(error.value) == message


def _write_large_pool(directory: Path) -> tuple[Path, Path]:
    with _candidate_file().open(encoding="utf-8", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    rows = []
    index = 0
    for decision in ("yes", "no"):
        for i in range(25):
            for j in range(25):
                candidate = candidates[index % len(candidates)]
                lat = -48.0 + 4.0 * i + (0.5 if decision == "no" else 0.0)
                lon = -168.0 + 14.0 * j + (0.5 if decision == "no" else 0.0)
                sentence = f"large pool sentence {index} {decision}"
                rows.append(
                    {
                        "source_pbf": "test-latest.osm.pbf",
                        "osm_type": "way",
                        "osm_id": index + 1,
                        "description_identity": f"description-{index}",
                        "tag_key": "description",
                        "sentence_index": 0,
                        "sentence": sentence,
                        "text_sha256": hashlib.sha256(sentence.encode()).hexdigest(),
                        "language_code": "eng",
                        "eunis_code": candidate["eunis_code"],
                        "eunis_name": candidate["eunis_name"],
                        "eunis_overlap_percentage": 80.0,
                        "eunis_source_version": "maps-v1",
                        "decision": decision,
                        "bbox_min_x": lon - 0.01,
                        "bbox_min_y": lat - 0.01,
                        "bbox_max_x": lon + 0.01,
                        "bbox_max_y": lat + 0.01,
                    }
                )
                index += 1
    labelled = directory / "labelled-eunis.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=LABELLED_SCHEMA), labelled)
    manifest = directory / "pipeline-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "computation_commit": "a" * 40,
                "validation_commit": "b" * 40,
                "input_snapshots": {"shared_input_revision": "fixture-revision"},
                "artifact_sha256": {"labelled-eunis.parquet": runner.sha256_file(labelled)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return labelled, manifest


def _existing_row(
    index: int, decision: str, cell: str, lat: float, *, text_sha256: str | None = None
) -> dict[str, Any]:
    row = _row(index, decision, lat, 0.0)
    if text_sha256 is not None:
        row["text_sha256"] = text_sha256
    return {
        **row,
        "h3_cell": cell,
        "cell_centre_lat": float(cell),
        "cell_centre_lon": 0.0,
        "sample_id": runner.sample_id_of(row),
    }


def test_extension_preserves_base_and_is_geographic_unique_and_deterministic() -> None:
    rows = _grid_rows()
    base = select_geographic_sample(
        rows, cell_of=h3_cell_of, centre_of=h3_centre_of, per_group=5, seed=42
    )

    added = extend_geographic_sample(
        rows,
        cell_of=h3_cell_of,
        centre_of=h3_centre_of,
        existing=base,
        per_group=12,
        seed=42,
    )
    repeated = extend_geographic_sample(
        rows,
        cell_of=h3_cell_of,
        centre_of=h3_centre_of,
        existing=base,
        per_group=12,
        seed=42,
    )
    selected = [*base, *added]

    assert repeated == added
    assert selected[: len(base)] == base
    for decision in ("yes", "no"):
        decision_rows = [row for row in selected if row["decision"] == decision]
        assert len(decision_rows) == 12
        assert len({row["h3_cell"] for row in decision_rows}) == 12
    assert len({row["text_sha256"] for row in selected}) == len(selected)
    assert len({(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in selected}) == len(
        selected
    )


@pytest.mark.parametrize("duplicate_kind", ["sentence", "polygon"])
def test_extension_never_reuses_base_sentence_or_polygon(
    duplicate_kind: str,
) -> None:
    base_yes = _existing_row(1, "yes", "0", 0.0)
    base_no = _existing_row(2, "no", "1", 1.0)
    duplicate = _row(3, "yes", 10.0, 0.0)
    if duplicate_kind == "sentence":
        duplicate["text_sha256"] = base_yes["text_sha256"]
    else:
        duplicate["osm_id"] = base_yes["osm_id"]
    distinct_yes = _row(4, "yes", 20.0, 0.0)
    distinct_no = _row(5, "no", 21.0, 0.0)

    added = extend_geographic_sample(
        [duplicate, distinct_yes, distinct_no],
        cell_of=_simple_cell_of,
        centre_of=_simple_centre_of,
        existing=[base_yes, base_no],
        per_group=2,
        seed=42,
    )

    assert all(row["text_sha256"] != base_yes["text_sha256"] for row in added)
    assert all(row["osm_id"] != base_yes["osm_id"] for row in added)


def test_extension_fails_if_unique_rows_cannot_fill_both_groups() -> None:
    base_yes = _existing_row(1, "yes", "0", 0.0)
    base_no = _existing_row(2, "no", "1", 1.0)
    duplicate = _row(3, "yes", 10.0, 0.0)
    duplicate["text_sha256"] = base_yes["text_sha256"]
    distinct_no = _row(5, "no", 21.0, 0.0)

    _assert_value_error(
        "need 1 more yes cells, found 0",
        lambda: extend_geographic_sample(
            [duplicate, distinct_no],
            cell_of=_simple_cell_of,
            centre_of=_simple_centre_of,
            existing=[base_yes, base_no],
            per_group=2,
            seed=42,
        ),
    )


@pytest.mark.parametrize(
    ("existing", "per_group", "error"),
    [
        ([{"decision": "maybe"}], 2, "existing sample has an unknown decision"),
        (
            [_existing_row(1, "yes", "0", 0.0), _existing_row(2, "yes", "1", 1.0)],
            1,
            "existing sample has more than 1 yes rows",
        ),
        (
            [_existing_row(1, "yes", "0", 0.0), _existing_row(2, "yes", "0", 0.0)],
            2,
            "existing sample has duplicate yes cells",
        ),
        (
            [
                _existing_row(1, "yes", "0", 0.0),
                _existing_row(2, "no", "1", 1.0, text_sha256=_row(1, "yes", 0, 0)["text_sha256"]),
            ],
            2,
            "existing sample must have unique sentence hashes and polygons",
        ),
    ],
)
def test_extension_rejects_invalid_existing_sample(
    existing: list[dict[str, Any]], per_group: int, error: str
) -> None:
    _assert_value_error(
        error,
        lambda: extend_geographic_sample(
            [],
            cell_of=_simple_cell_of,
            centre_of=_simple_centre_of,
            existing=existing,
            per_group=per_group,
            seed=42,
        ),
    )


def test_extension_allows_yes_and_no_to_share_a_new_cell() -> None:
    base = [_existing_row(1, "yes", "0", 0.0), _existing_row(2, "no", "1", 1.0)]
    rows = [_row(3, "yes", 20.0, 0.0), _row(4, "no", 20.0, 0.5)]

    added = extend_geographic_sample(
        rows,
        cell_of=_simple_cell_of,
        centre_of=_simple_centre_of,
        existing=base,
        per_group=2,
        seed=42,
    )

    assert {row["h3_cell"] for row in added} == {"20"}


def test_base_reader_preserves_the_committed_100_rows_and_checks_integrity(
    tmp_path: Path,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))

    rows = runner._read_base_sample(
        BASE_SAMPLE,
        per_group=500,
        seed=42,
        expected_labelled_sha256=sample["source"]["sha256"],
        expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
    )

    assert rows == sample["selected_rows"]
    assert len(rows) == 100
    assert (
        sample["selection"]["sample_ids_sha256"]
        == "e481f7fa38a6efaccdf2358a9bdb4b2b3a34df8cd85c7023df74a12a44dddc6d"
    )

    altered = json.loads(json.dumps(sample))
    altered["selected_rows"][0]["sample_id"] = "0" * 64
    altered_path = tmp_path / "altered.json"
    altered_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen sample ID checksum mismatch"):
        runner._read_base_sample(
            altered_path,
            per_group=500,
            seed=42,
            expected_labelled_sha256=sample["source"]["sha256"],
            expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
        )


def test_base_reader_requires_the_same_pipeline_manifest() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="pipeline manifest must have the same SHA-256"):
        runner._read_base_sample(
            BASE_SAMPLE,
            per_group=500,
            seed=42,
            expected_labelled_sha256=sample["source"]["sha256"],
            expected_pipeline_manifest_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("method", "unsupported", "frozen sample selection method is not supported"),
        (
            "sample_size",
            999,
            "frozen sample size does not match its per-group count",
        ),
        ("h3_resolution", 4, "frozen sample H3 resolution does not match the protocol"),
        ("cell_count", 0, "frozen sample cell count does not match its rows"),
        ("decision_counts", {"no": 50}, "frozen sample counts do not match its rows"),
        (
            "unique_sentence_hash_count",
            99,
            "frozen sample counts do not match its rows",
        ),
        (
            "unique_polygon_count",
            99,
            "frozen sample counts do not match its rows",
        ),
    ],
)
def test_selection_metadata_rejects_protocol_drift(field: str, value: Any, error: str) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    selection = {**sample["selection"], field: value}

    _assert_value_error(
        error,
        lambda: runner._validate_selection_metadata(sample["selected_rows"], selection),
    )


def test_selection_cell_validation_keeps_the_two_protocols_distinct() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    rows = [dict(row) for row in sample["selected_rows"]]
    yes_rows = [row for row in rows if row["decision"] == "yes"]
    yes_rows[1]["h3_cell"] = yes_rows[0]["h3_cell"]
    _assert_value_error(
        "frozen sample must have one sentence per cell within each group",
        lambda: runner._require_unique_group_cells(rows, runner.SELECTION_METHOD),
    )

    rows = [
        {"decision": "yes", "h3_cell": "shared"},
        {"decision": "no", "h3_cell": "shared"},
    ]
    runner._require_unique_group_cells(rows, runner.EXTENDED_SELECTION_METHOD)
    _assert_value_error(
        "published pilot sample must use disjoint yes and no cells",
        lambda: runner._require_unique_group_cells(rows, runner.SELECTION_METHOD),
    )


def test_freeze_extension_keeps_the_committed_base_order_and_writes_1000_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labelled, pipeline_manifest = _write_large_pool(tmp_path)
    base_dir = tmp_path / "pilot-100-seed42"
    extended_dir = tmp_path / "pilot-1000-seed42"
    base_summary = runner.freeze_sample(
        labelled, _candidate_file(), pipeline_manifest, base_dir, per_group=50, seed=42
    )
    base_sample = json.loads((base_dir / "frozen_sample.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(
        runner,
        "EXPECTED_SAMPLE_IDS_SHA256",
        base_sample["selection"]["sample_ids_sha256"],
    )

    extended_summary = runner.freeze_sample(
        labelled,
        _candidate_file(),
        pipeline_manifest,
        extended_dir,
        per_group=500,
        seed=42,
        extend_from=base_dir / "frozen_sample.json",
    )
    extended_sample, rows, candidates = runner.read_frozen_pilot_inputs(extended_dir)

    assert base_summary["sample_size"] == 100
    assert extended_summary["sample_size"] == 1000
    assert rows[:100] == base_sample["selected_rows"]
    assert len(rows) == 1000
    assert len(candidates) == 158
    assert extended_sample["selection"]["method"] == runner.EXTENDED_SELECTION_METHOD
    assert extended_sample["selection"]["decision_counts"] == {"no": 500, "yes": 500}


@pytest.mark.parametrize(
    ("selection_update", "requested_per_group", "error"),
    [
        ({"seed": 7}, 500, "base sample must be the published pilot sample with the same seed"),
        (
            {"method": "unsupported"},
            500,
            "base sample must be the published pilot sample with the same seed",
        ),
        (
            {"sample_size": 99},
            500,
            "base sample must be the published pilot sample with the same seed",
        ),
        (
            {"sample_ids_sha256": "0" * 64},
            500,
            "base sample must be the published pilot sample with the same seed",
        ),
        ({"per_group": 49}, 500, "base sample must contain the published 50/50 sample"),
        ({}, 49, "requested per_group is smaller than the base sample"),
    ],
)
def test_base_reader_rejects_a_different_seed_or_smaller_target(
    tmp_path: Path,
    selection_update: dict[str, Any],
    requested_per_group: int,
    error: str,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    sample["selection"].update(selection_update)
    altered_path = tmp_path / "altered.json"
    altered_path.write_text(json.dumps(sample), encoding="utf-8")

    _assert_value_error(
        error,
        lambda: runner._read_base_sample(
            altered_path,
            per_group=requested_per_group,
            seed=42,
            expected_labelled_sha256=sample["source"]["sha256"],
            expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
        ),
    )


def test_extension_helpers_keep_seeded_maximin_distances() -> None:
    centres = {"c0": (0.0, 0.0), "c1": (0.0, 2.0), "anchor": (0.0, 1.0)}
    nearest = geo_sampling._nearest_extension_distances(["c0", "c1"], [], centres)
    assert all(math.isinf(distance) for distance in nearest.values())

    geo_sampling._update_nearest_extension_distances(["c0", "c1"], nearest, centres, "anchor")
    assert nearest == {
        "c0": geo_sampling._haversine_km(centres["c0"], centres["anchor"]),
        "c1": geo_sampling._haversine_km(centres["c1"], centres["anchor"]),
    }
    assert geo_sampling._farthest_extension_cell(["c2", "c1"], {"c1": 1.0, "c2": 1.0}, 42) == "c1"

    rows = {
        "c1": [_row(11, "yes", 1.0, 0.0)],
        "c2": [_row(12, "yes", 2.0, 0.0)],
    }
    selected = geo_sampling._extend_group(
        "yes",
        rows,
        [],
        1,
        lambda cell: (float(cell[1:]), 0.0),
        seed=42,
    )
    assert [row["h3_cell"] for row in selected] == ["c1"]


def test_extension_default_seed_and_seeded_duplicate_winner() -> None:
    rows = [
        _row(1, "yes", 0.0, 0.0),
        _row(2, "yes", 2.0, 0.0),
        _row(3, "no", 10.0, 0.0),
    ]
    default_seed = extend_geographic_sample(
        rows,
        cell_of=_simple_cell_of,
        centre_of=_simple_centre_of,
        existing=[],
        per_group=1,
    )
    explicit_seed = extend_geographic_sample(
        rows,
        cell_of=_simple_cell_of,
        centre_of=_simple_centre_of,
        existing=[],
        per_group=1,
        seed=42,
    )
    assert default_seed == explicit_seed
    assert default_seed[0]["h3_cell"] == "2"

    first = _row(21, "yes", 0.0, 0.0)
    duplicate = _row(22, "yes", 1.0, 0.0)
    duplicate["text_sha256"] = first["text_sha256"]
    assert [row["osm_id"] for row in geo_sampling._deduplicate([first, duplicate], 42)] == [21]


def test_existing_extension_cells_are_partitioned_by_decision() -> None:
    existing = [_existing_row(1, "yes", "1", 1.0), _existing_row(2, "no", "2", 2.0)]
    assert geo_sampling._chosen_extension_cells("yes", existing) == ["1"]
    assert geo_sampling._chosen_extension_cells("no", existing) == ["2"]


def test_extension_requires_a_positive_target_and_base_rows_are_grouped_correctly() -> None:
    _assert_value_error(
        "per_group must be positive",
        lambda: extend_geographic_sample(
            [],
            cell_of=_simple_cell_of,
            centre_of=_simple_centre_of,
            existing=[],
            per_group=0,
        ),
    )
    rows = [
        {"decision": "yes", "h3_cell": "yes-cell"},
        {"decision": "no", "h3_cell": "no-cell"},
    ]
    assert runner._cells_by_decision(rows) == {"yes": ["yes-cell"], "no": ["no-cell"]}


def test_base_sample_reader_rejects_malformed_rows_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    malformed_rows = {**sample, "selected_rows": ["not a row object"]}
    malformed_rows_path = tmp_path / "malformed-rows.json"
    malformed_rows_path.write_text(json.dumps(malformed_rows), encoding="utf-8")
    _assert_value_error(
        "base frozen sample must contain selected row objects",
        lambda: runner._read_base_sample(
            malformed_rows_path,
            per_group=500,
            seed=42,
            expected_labelled_sha256=sample["source"]["sha256"],
            expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
        ),
    )

    malformed_json = tmp_path / "malformed-json.json"
    malformed_json.write_text("{", encoding="utf-8")
    _assert_value_error(
        "base frozen sample is unreadable",
        lambda: runner._read_base_sample_json(malformed_json),
    )
    missing_selection = tmp_path / "missing-selection.json"
    missing_selection.write_text("[]", encoding="utf-8")
    _assert_value_error(
        "base frozen sample must contain a selection object",
        lambda: runner._read_base_sample_json(missing_selection),
    )

    original_read_text = Path.read_text
    encodings: list[str | None] = []

    def record_encoding(path: Path, *args: Any, **kwargs: Any) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_encoding)
    runner._read_base_sample_json(BASE_SAMPLE)
    assert encodings == ["utf-8"]


def test_selection_validation_checks_exact_integer_protocol_values() -> None:
    _assert_value_error(
        "frozen sample size does not match its per-group count",
        lambda: runner._require_declared_group_size({"sample_size": 100}),
    )
    _assert_value_error(
        "frozen sample size does not match its per-group count",
        lambda: runner._require_declared_group_size({"sample_size": 100, "per_group": 50.0}),
    )
    _assert_value_error(
        "base sample must contain the published 50/50 sample",
        lambda: runner._validate_base_group_size({"per_group": 50.0}),
    )

    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    runner._validate_base_selection(sample["selection"], per_group=50, seed=42)


def test_base_reader_reports_exact_source_provenance_errors() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    _assert_value_error(
        "base frozen sample must contain source provenance",
        lambda: runner._validate_base_source({}, "a" * 64, "b" * 64),
    )
    _assert_value_error(
        "base sample and labelled pool must have the same SHA-256",
        lambda: runner._validate_base_source(
            sample,
            "0" * 64,
            sample["source"]["pipeline_manifest_sha256"],
        ),
    )
    _assert_value_error(
        "base sample and pipeline manifest must have the same SHA-256",
        lambda: runner._validate_base_source(
            sample,
            sample["source"]["sha256"],
            "0" * 64,
        ),
    )


def test_frozen_selection_enforces_disjoint_cells_for_the_published_protocol() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    rows = [dict(row) for row in sample["selected_rows"]]
    yes = next(row for row in rows if row["decision"] == "yes")
    no = next(row for row in rows if row["decision"] == "no")
    no["h3_cell"] = yes["h3_cell"]
    no["cell_centre_lat"] = yes["cell_centre_lat"]
    no["cell_centre_lon"] = yes["cell_centre_lon"]
    selection = {
        **sample["selection"],
        "cell_count": len({row["h3_cell"] for row in rows}),
    }
    _assert_value_error(
        "published pilot sample must use disjoint yes and no cells",
        lambda: runner._validate_frozen_selection(rows, selection),
    )


def test_dspark_selection_gate_rejects_unsupported_or_malformed_sizes() -> None:
    _assert_value_error(
        "DSpark evaluation requires a supported pilot sample size",
        lambda: dspark_runner._validate_sample_selection(
            {
                "sample_size": 99,
                "seed": 42,
                "per_group": 49,
                "method": runner.EXTENDED_SELECTION_METHOD,
            },
            [{}] * 99,
        ),
    )
    _assert_value_error(
        "DSpark evaluation requires a supported pilot sample size",
        lambda: dspark_runner._validate_sample_selection(
            {"sample_size": 100, "seed": 42}, [{}] * 99
        ),
    )
    for selection in (
        {"per_group": 499, "method": runner.EXTENDED_SELECTION_METHOD},
        {"per_group": 500.0, "method": runner.EXTENDED_SELECTION_METHOD},
        {"per_group": 500, "method": "wrong"},
    ):
        _assert_value_error(
            "frozen sample selection method or group size does not match the protocol",
            lambda selection=selection: dspark_runner._validate_sample_selection_protocol(
                selection, 1000
            ),
        )


def test_select_frozen_rows_forwards_seed_to_geographic_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    calls: list[int] = []

    def record_seed(*_args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs["seed"])
        return []

    monkeypatch.setattr(runner, "extend_geographic_sample", record_seed)
    rows, method = runner._select_frozen_rows(
        [],
        500,
        42,
        BASE_SAMPLE,
        sample["source"]["sha256"],
        sample["source"]["pipeline_manifest_sha256"],
    )
    assert rows == sample["selected_rows"]
    assert method == runner.EXTENDED_SELECTION_METHOD
    assert calls == [42]


def test_dspark_runner_passes_sample_size_when_choosing_default_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = {"selection": {"sample_size": 1000}}
    sample_rows = [{}] * 1000
    output_sizes: list[int] = []

    class StopAfterOutputSelection(Exception):
        pass

    def choose_output(
        _run_dir: Path, output_dir: Path | None, _smoke: bool, sample_size: int = 100
    ) -> Path:
        output_sizes.append(sample_size)
        return output_dir or tmp_path / f"pilot-{sample_size}"

    monkeypatch.setattr(
        dspark_runner,
        "read_frozen_pilot_inputs",
        lambda _run_dir: (sample, sample_rows, []),
    )
    monkeypatch.setattr(dspark_runner, "_validate_published_sample", lambda *_args: None)
    monkeypatch.setattr(dspark_runner, "_output_destination", choose_output)
    monkeypatch.setattr(
        dspark_runner,
        "_validate_run_options",
        lambda *_args: (_ for _ in ()).throw(StopAfterOutputSelection()),
    )

    with pytest.raises(StopAfterOutputSelection):
        dspark_runner.run_dspark_pilot(
            tmp_path / "frozen",
            computation_commit="a" * 40,
            validation_commit="b" * 40,
            output_dir=tmp_path / "output",
        )
    assert output_sizes == [1000]
