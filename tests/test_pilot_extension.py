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
BASE_MANIFEST = BASE_SAMPLE.with_name("manifest.json")


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
                # Both groups share the same geographic candidate capacity, which makes the
                # common regional quota contract feasible while keeping row identities unique.
                lat = -48.0 + 4.0 * i
                lon = -168.0 + 14.0 * j
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


def test_extension_uses_stored_base_centres_as_maximin_anchors() -> None:
    base = [_existing_row(1, "yes", "1", 1.0), _existing_row(2, "no", "2", 2.0)]

    def centre_of(cell: str) -> tuple[float, float]:
        if cell in {"1", "2"}:
            raise AssertionError("the published base centre must not be recomputed")
        return float(cell), 0.0

    added = extend_geographic_sample(
        [_row(3, "yes", 10.0, 0.0), _row(4, "no", 20.0, 0.0)],
        cell_of=_simple_cell_of,
        centre_of=centre_of,
        existing=base,
        per_group=2,
        seed=42,
    )

    assert [row["h3_cell"] for row in added] == ["10", "20"]


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
    assert runner._sha256_json(rows) == runner.EXPECTED_BASE_SELECTED_ROWS_SHA256
    assert (
        sample["selection"]["sample_ids_sha256"]
        == "e481f7fa38a6efaccdf2358a9bdb4b2b3a34df8cd85c7023df74a12a44dddc6d"
    )

    altered = json.loads(json.dumps(sample))
    altered["selected_rows"][0]["sample_id"] = "0" * 64
    altered_path = tmp_path / "altered.json"
    altered_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(
        ValueError, match="base frozen sample checksum does not match the published pilot"
    ):
        runner._read_base_sample(
            altered_path,
            per_group=500,
            seed=42,
            expected_labelled_sha256=sample["source"]["sha256"],
            expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
        )


def test_pinned_base_reader_does_not_recompute_the_published_cell_centres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    expected_rows = sample["selected_rows"]
    row_49 = expected_rows[49]
    original_centre = runner.h3_centre_of

    def changed_platform_centre(cell: str) -> tuple[float, float]:
        lat, lon = original_centre(cell)
        if cell == row_49["h3_cell"]:
            return lat, math.nextafter(lon, math.inf)
        return lat, lon

    monkeypatch.setattr(runner, "h3_centre_of", changed_platform_centre)
    rows = runner._read_base_sample(
        BASE_SAMPLE,
        per_group=500,
        seed=42,
        expected_labelled_sha256=sample["source"]["sha256"],
        expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
    )
    _, run_rows = runner._read_frozen_sample(BASE_SAMPLE.parent)

    assert rows == expected_rows
    assert run_rows == expected_rows


def test_frozen_row_validation_rejects_a_sentence_that_disagrees_with_its_stored_hash() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    sample["selected_rows"][0]["sentence"] = "changed sentence with the original stored hash"

    rows = sample["selected_rows"]
    _assert_value_error(
        "frozen sample sentence hash mismatch",
        lambda: runner._validate_frozen_rows(rows, sample["selection"]),
    )


def test_base_reader_rejects_a_non_text_sentence() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    rows = [dict(row) for row in sample["selected_rows"]]
    rows[0]["sentence"] = None

    _assert_value_error(
        "frozen sample sentence must be text",
        lambda: runner._validate_frozen_rows(rows, sample["selection"]),
    )


def test_frozen_row_validation_recomputes_ids_from_row_identity() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    rows = [dict(row) for row in sample["selected_rows"]]
    rows[0]["osm_id"] += 1

    _assert_value_error(
        "frozen sample ID does not match its row identity",
        lambda: runner._validate_frozen_rows(rows, sample["selection"]),
    )


def test_base_reader_rejects_an_incomplete_row_identity() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    rows = [dict(row) for row in sample["selected_rows"]]
    del rows[0]["osm_id"]

    _assert_value_error(
        "frozen sample row identity is incomplete",
        lambda: runner._validate_frozen_rows(rows, sample["selection"]),
    )


def test_base_reader_rejects_modified_gold_when_manifest_output_hash_is_rewritten(
    tmp_path: Path,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    with _candidate_file().open(encoding="utf-8", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    changed = next(
        row for row in candidates if row["eunis_code"] != sample["selected_rows"][0]["eunis_code"]
    )
    sample["selected_rows"][0]["eunis_code"] = changed["eunis_code"]
    sample["selected_rows"][0]["eunis_name"] = changed["eunis_name"]

    base_dir = tmp_path / "tampered-base"
    base_dir.mkdir()
    sample_path = base_dir / "frozen_sample.json"
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    manifest["outputs_sha256"]["frozen_sample.json"] = runner.sha256_file(sample_path)
    (base_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="base frozen sample checksum does not match the published pilot",
    ):
        runner._read_base_sample(
            sample_path,
            per_group=500,
            seed=42,
            expected_labelled_sha256=sample["source"]["sha256"],
            expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
        )


def test_base_reader_requires_a_manifest_that_binds_the_committed_sample(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "base-without-manifest"
    base_dir.mkdir()
    sample_path = base_dir / "frozen_sample.json"
    sample_path.write_bytes(BASE_SAMPLE.read_bytes())
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))

    _assert_value_error(
        "base frozen manifest.json is required to verify the published sample",
        lambda: runner._read_base_sample(
            sample_path,
            per_group=500,
            seed=42,
            expected_labelled_sha256=sample["source"]["sha256"],
            expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
        ),
    )


@pytest.mark.parametrize(
    ("field_group", "field", "value"),
    [
        ("outputs_sha256", "frozen_sample.json", "0" * 64),
        ("outputs_sha256", "candidate_labels.csv", "0" * 64),
        ("source_sha256", "labelled-eunis.parquet", "0" * 64),
        ("source_sha256", "pipeline-manifest.json", "0" * 64),
        ("selection", "per_group", 49),
    ],
)
def test_base_reader_rejects_a_manifest_that_does_not_bind_its_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_group: str,
    field: str,
    value: Any,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    if field_group == "selection":
        manifest[field] = value
    else:
        manifest[field_group][field] = value
    base_dir = tmp_path / "rewritten-base"
    base_dir.mkdir()
    sample_path = base_dir / "frozen_sample.json"
    sample_path.write_bytes(BASE_SAMPLE.read_bytes())
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_BASE_MANIFEST_SHA256", runner.sha256_file(manifest_path))

    with pytest.raises(ValueError, match="base frozen manifest"):
        runner._read_base_sample(
            sample_path,
            per_group=500,
            seed=42,
            expected_labelled_sha256=sample["source"]["sha256"],
            expected_pipeline_manifest_sha256=sample["source"]["pipeline_manifest_sha256"],
        )


@pytest.mark.parametrize("payload", ["not-json", "[]"])
def test_base_reader_rejects_an_unreadable_or_non_object_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    base_dir = tmp_path / "invalid-base-manifest"
    base_dir.mkdir()
    sample_path = base_dir / "frozen_sample.json"
    sample_path.write_bytes(BASE_SAMPLE.read_bytes())
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_BASE_MANIFEST_SHA256", runner.sha256_file(manifest_path))

    with pytest.raises(ValueError, match="base frozen manifest"):
        runner._read_base_sample(
            sample_path,
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


def test_m49_metadata_builder_rejects_counts_that_change_after_selection() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))

    _assert_value_error(
        "M49 selected region counts changed during manifest construction",
        lambda: runner._extension_geographic_metadata(
            sample["selected_rows"], {"selected_counts_by_decision": {}}
        ),
    )


def test_m49_assignment_counter_rejects_an_empty_stratum_or_status() -> None:
    with pytest.raises(ValueError, match="M49 assignment fields are invalid"):
        runner._m49_assignment_counts(
            [{"decision": "yes"}], [runner.M49Assignment("", None, None, "")]
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


def test_common_m49_quota_validation_allows_zero_quota_one_sided_regions() -> None:
    runner._require_common_m49_group_counts(
        {"034": 0, "039": 1},
        {"yes": {"039": 1}, "no": {"039": 1}},
    )


def test_m49_unique_cell_capacity_requires_memberships_for_every_reported_cell() -> None:
    _assert_value_error(
        "extended frozen sample M49 unique cell capacities are invalid",
        lambda: runner._validate_m49_group_cell_capacity(
            "yes",
            {"yes": 2},
            {"yes": 0},
            {"yes": {"039": 1}},
        ),
    )


def test_m49_unique_cell_capacity_rejects_too_many_overlapping_memberships() -> None:
    _assert_value_error(
        "extended frozen sample M49 unique cell capacities are invalid",
        lambda: runner._validate_m49_group_cell_capacity(
            "yes",
            {"yes": 2},
            {"yes": 1},
            {"yes": {"039": 2, "151": 2, "154": 2}},
        ),
    )


def test_m49_quota_bounds_reject_a_quota_above_cell_capacity() -> None:
    _assert_value_error(
        "extended frozen sample M49 quota exceeds its recorded capacity",
        lambda: runner._validate_m49_quota_bounds(
            {"039": 2},
            {"yes": {"039": 0}, "no": {"039": 0}},
            {"yes": {"039": 1}, "no": {"039": 1}},
        ),
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
    monkeypatch.setattr(
        runner,
        "EXPECTED_BASE_FROZEN_SAMPLE_SHA256",
        runner.sha256_file(base_dir / "frozen_sample.json"),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_BASE_MANIFEST_SHA256",
        runner.sha256_file(base_dir / "manifest.json"),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_BASE_SELECTED_ROWS_SHA256",
        runner._sha256_json(base_sample["selected_rows"]),
    )
    original_centre = runner.h3_centre_of
    changed_base_cell = base_sample["selected_rows"][49]["h3_cell"]

    def changed_platform_centre(cell: str) -> tuple[float, float]:
        lat, lon = original_centre(cell)
        if cell == changed_base_cell:
            return lat, math.nextafter(lon, math.inf)
        return lat, lon

    monkeypatch.setattr(runner, "h3_centre_of", changed_platform_centre)

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
    geography = extended_sample["selection"]["geographic_stratification"]
    assert geography["protocol_version"] == "un-m49-10m-common-quota-cell-flow-maximin-v2"
    assert (
        geography["selected_counts_by_decision"]["yes"]
        == geography["selected_counts_by_decision"]["no"]
        == geography["target_counts_per_group"]
    )
    output_manifest = json.loads((extended_dir / "manifest.json").read_text(encoding="utf-8"))
    assert output_manifest["geographic_stratification"] == geography

    changed_prefix = [dict(row) for row in rows]
    changed_prefix[49]["cell_centre_lon"] = math.nextafter(
        changed_prefix[49]["cell_centre_lon"], math.inf
    )
    _assert_value_error(
        "extended frozen sample base rows do not match the published pilot",
        lambda: runner._validate_frozen_rows(changed_prefix, extended_sample["selection"]),
    )

    changed_extension = [dict(row) for row in rows]
    changed_extension[100]["cell_centre_lon"] = math.nextafter(
        changed_extension[100]["cell_centre_lon"], math.inf
    )
    _assert_value_error(
        "frozen sample H3 cell does not match its centre",
        lambda: runner._validate_frozen_rows(changed_extension, extended_sample["selection"]),
    )

    assignments_changed = json.loads(json.dumps(extended_sample["selection"]))
    assigned_index = next(
        index
        for index, item in enumerate(
            assignments_changed["geographic_stratification"]["assignments_by_sample_id"]
        )
        if item["status"] == "assigned"
    )
    assignments_changed["geographic_stratification"]["assignments_by_sample_id"][assigned_index][
        "stratum"
    ] = "UNASSIGNED"
    _assert_value_error(
        "extended frozen sample M49 assignment does not match its coordinates",
        lambda: runner._validate_frozen_rows(rows, assignments_changed),
    )

    capacity_changed = json.loads(json.dumps(extended_sample["selection"]))
    geography = capacity_changed["geographic_stratification"]
    stratum = next(
        region
        for region, quota in geography["target_counts_per_group"].items()
        if quota
        > max(
            geography["prefix_counts_by_decision"][group].get(region, 0) for group in ("yes", "no")
        )
    )
    geography["candidate_cell_capacity_by_decision"]["yes"][stratum] = 0
    _assert_value_error(
        "extended frozen sample M49 unique cell capacities are invalid",
        lambda: runner._validate_frozen_rows(rows, capacity_changed),
    )

    missing_assignments = json.loads(json.dumps(extended_sample["selection"]))
    missing_assignments["geographic_stratification"]["assignments_by_sample_id"].pop()
    _assert_value_error(
        "extended frozen sample M49 assignments do not cover every row",
        lambda: runner._validate_frozen_rows(rows, missing_assignments),
    )

    assignments_reordered = json.loads(json.dumps(extended_sample["selection"]))
    assignments_reordered["geographic_stratification"]["assignments_by_sample_id"][0][
        "sample_id"
    ] = "not-the-row-id"
    _assert_value_error(
        "extended frozen sample M49 assignment order does not match its rows",
        lambda: runner._validate_frozen_rows(rows, assignments_reordered),
    )

    status_counts_changed = json.loads(json.dumps(extended_sample["selection"]))
    status_counts_changed["geographic_stratification"]["assignment_status_counts_by_decision"][
        "yes"
    ]["assigned"] = -1
    _assert_value_error(
        "extended frozen sample M49 status counts do not match its rows",
        lambda: runner._validate_frozen_rows(rows, status_counts_changed),
    )

    region_names_changed = json.loads(json.dumps(extended_sample["selection"]))
    geography = region_names_changed["geographic_stratification"]
    named_region = next(code for code in geography["region_names_by_code"] if code != "UNASSIGNED")
    geography["region_names_by_code"][named_region] = "incorrect region"
    _assert_value_error(
        "extended frozen sample M49 region names do not match its rows",
        lambda: runner._validate_frozen_rows(rows, region_names_changed),
    )

    selected_counts_changed = json.loads(json.dumps(extended_sample["selection"]))
    selected_counts_changed["geographic_stratification"]["selected_counts_by_decision"]["yes"][
        named_region
    ] = -1
    _assert_value_error(
        "extended frozen sample M49 selected counts do not match its rows",
        lambda: runner._validate_frozen_rows(rows, selected_counts_changed),
    )

    quota_changed = json.loads(json.dumps(extended_sample["selection"]))
    quota_changed["geographic_stratification"]["target_counts_per_group"][named_region] = 0
    _assert_value_error(
        "extended frozen sample does not meet its common M49 quota vector",
        lambda: runner._validate_frozen_rows(rows, quota_changed),
    )

    quota_record_missing = json.loads(json.dumps(extended_sample["selection"]))
    quota_record_missing["geographic_stratification"]["target_counts_per_group"] = []
    _assert_value_error(
        "extended frozen sample M49 quota records are missing",
        lambda: runner._validate_frozen_rows(rows, quota_record_missing),
    )

    prefix_counts_changed = json.loads(json.dumps(extended_sample["selection"]))
    prefix_counts_changed["geographic_stratification"]["prefix_counts_by_decision"]["yes"][
        named_region
    ] = (
        prefix_counts_changed["geographic_stratification"]["prefix_counts_by_decision"]["yes"].get(
            named_region, 0
        )
        + 1
    )
    _assert_value_error(
        "extended frozen sample M49 prefix counts do not match its base rows",
        lambda: runner._validate_frozen_rows(rows, prefix_counts_changed),
    )

    eligible_regions_changed = json.loads(json.dumps(extended_sample["selection"]))
    eligible_regions_changed["geographic_stratification"]["eligible_subregion_codes"] = []
    _assert_value_error(
        "extended frozen sample M49 eligible region list is inconsistent",
        lambda: runner._validate_frozen_rows(rows, eligible_regions_changed),
    )

    capacity_shape_changed = json.loads(json.dumps(extended_sample["selection"]))
    capacity_shape_changed["geographic_stratification"]["candidate_cell_capacity_by_decision"][
        "yes"
    ] = []
    _assert_value_error(
        "extended frozen sample M49 capacity records are invalid",
        lambda: runner._validate_frozen_rows(rows, capacity_shape_changed),
    )

    capacity_value_changed = json.loads(json.dumps(extended_sample["selection"]))
    capacity_value_changed["geographic_stratification"]["candidate_cell_capacity_by_decision"][
        "yes"
    ][named_region] = -1
    _assert_value_error(
        "extended frozen sample M49 capacities must be non-negative integers",
        lambda: runner._validate_frozen_rows(rows, capacity_value_changed),
    )

    provenance_changed = json.loads(json.dumps(extended_sample["selection"]))
    provenance_changed["geographic_stratification"]["geometry_asset"]["sha256"] = "0" * 64
    _assert_value_error(
        "extended frozen sample M49 boundary provenance is inconsistent",
        lambda: runner._validate_frozen_rows(rows, provenance_changed),
    )

    quota_method_changed = json.loads(json.dumps(extended_sample["selection"]))
    quota_method_changed["geographic_stratification"]["quota_method"] = "unsupported"
    _assert_value_error(
        "extended frozen sample M49 quota method is unsupported",
        lambda: runner._validate_frozen_rows(rows, quota_method_changed),
    )


def test_extension_reconstructed_manifest_mode_keeps_strict_default_and_audits_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labelled, original_manifest = _write_large_pool(tmp_path)
    base_dir = tmp_path / "pilot-100-seed42"
    runner.freeze_sample(
        labelled, _candidate_file(), original_manifest, base_dir, per_group=50, seed=42
    )
    sample_path = base_dir / "frozen_sample.json"
    manifest_path = base_dir / "manifest.json"
    base_sample = json.loads(sample_path.read_text(encoding="utf-8"))
    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    base_sample["source"]["pipeline_manifest_sha256"] = (
        runner.EXPECTED_BASE_PIPELINE_MANIFEST_SHA256
    )
    sample_path.write_text(json.dumps(base_sample, sort_keys=True, indent=2) + "\n")
    base_manifest["source_sha256"][runner.PIPELINE_MANIFEST_NAME] = (
        runner.EXPECTED_BASE_PIPELINE_MANIFEST_SHA256
    )
    base_manifest["outputs_sha256"]["frozen_sample.json"] = runner.sha256_file(sample_path)
    manifest_path.write_text(json.dumps(base_manifest, sort_keys=True, indent=2) + "\n")

    monkeypatch.setattr(
        runner, "EXPECTED_SAMPLE_IDS_SHA256", base_sample["selection"]["sample_ids_sha256"]
    )
    monkeypatch.setattr(
        runner, "EXPECTED_BASE_FROZEN_SAMPLE_SHA256", runner.sha256_file(sample_path)
    )
    monkeypatch.setattr(runner, "EXPECTED_BASE_MANIFEST_SHA256", runner.sha256_file(manifest_path))
    monkeypatch.setattr(
        runner,
        "EXPECTED_BASE_SELECTED_ROWS_SHA256",
        runner._sha256_json(base_sample["selected_rows"]),
    )
    monkeypatch.setattr(runner, "EXPECTED_LABELLED_POOL_SHA256", runner.sha256_file(labelled))

    reconstructed_manifest = tmp_path / "reconstructed-manifest.json"
    reconstructed_data = json.loads(original_manifest.read_text(encoding="utf-8"))
    reconstructed_data["reconstructed"] = True
    reconstructed_manifest.write_text(
        json.dumps(reconstructed_data, sort_keys=True), encoding="utf-8"
    )
    reconstructed_hash = runner.sha256_file(reconstructed_manifest)

    incorrect_manifest = tmp_path / "incorrect-manifest.json"
    incorrect_data = json.loads(reconstructed_manifest.read_text(encoding="utf-8"))
    incorrect_data["artifact_sha256"][runner.LABELLED_NAME] = "0" * 64
    incorrect_manifest.write_text(json.dumps(incorrect_data), encoding="utf-8")
    _assert_value_error(
        "labelled-eunis.parquet SHA-256 does not match the pipeline manifest",
        lambda: runner._validate_freeze_inputs(
            labelled, _candidate_file(), incorrect_manifest, runner.CANDIDATE_LABELS_SHA256
        ),
    )

    _assert_value_error(
        "base sample and pipeline manifest must have the same SHA-256",
        lambda: runner.freeze_sample(
            labelled,
            _candidate_file(),
            reconstructed_manifest,
            tmp_path / "strict-extension",
            per_group=500,
            seed=42,
            extend_from=sample_path,
        ),
    )

    extension_dir = tmp_path / "reconstructed-extension"
    runner.freeze_sample(
        labelled,
        _candidate_file(),
        reconstructed_manifest,
        extension_dir,
        per_group=500,
        seed=42,
        extend_from=sample_path,
        allow_reconstructed_pipeline_manifest=True,
    )
    extended, rows, _ = runner.read_frozen_pilot_inputs(extension_dir)
    provenance = extended["source"]["pipeline_manifest_provenance"]

    assert rows[:100] == base_sample["selected_rows"]
    assert len(rows) == 1000
    assert extended["selection"]["sample_ids_sha256"] == runner._sha256_json(
        [row["sample_id"] for row in rows]
    )
    assert extended["source"]["pipeline_manifest_sha256"] == reconstructed_hash
    assert provenance["historical_pipeline_manifest_sha256"] == (
        runner.EXPECTED_BASE_PIPELINE_MANIFEST_SHA256
    )
    assert provenance["extension_pipeline_manifest_sha256"] == reconstructed_hash
    assert provenance["labelled_pool_sha256"] == runner.sha256_file(labelled)
    assert provenance["base_frozen_sample_sha256"] == runner.sha256_file(sample_path)
    assert provenance["base_frozen_manifest_sha256"] == runner.sha256_file(manifest_path)
    assert provenance["base_selected_rows_sha256"] == runner._sha256_json(
        base_sample["selected_rows"]
    )


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


def test_extension_uses_caller_seed_when_deduplicating_duplicate_sentences() -> None:
    first = _row(1, "yes", 10.0, 0.0)
    duplicate = _row(2, "yes", 20.0, 0.0)
    third_duplicate = _row(6, "yes", 60.0, 0.0)
    duplicate["text_sha256"] = third_duplicate["text_sha256"] = first["text_sha256"]
    no_row = _row(10, "no", 30.0, 0.0)

    added = extend_geographic_sample(
        [first, duplicate, third_duplicate, no_row],
        cell_of=_simple_cell_of,
        centre_of=_simple_centre_of,
        existing=[],
        per_group=1,
        seed=7,
    )

    assert [(row["decision"], row["osm_id"]) for row in added] == [("yes", 2), ("no", 10)]


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
    for sample_size in (100.0, True):
        _assert_value_error(
            "frozen sample size does not match its per-group count",
            lambda sample_size=sample_size: runner._require_declared_group_size(
                {"sample_size": sample_size, "per_group": 50}
            ),
        )
    _assert_value_error(
        "base sample must contain the published 50/50 sample",
        lambda: runner._validate_base_group_size({"per_group": 50.0}),
    )

    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    runner._validate_base_selection(sample["selection"], per_group=50, seed=42)
    for field, value in (("seed", 42.0), ("sample_size", 100.0)):
        malformed = {**sample["selection"], field: value}
        _assert_value_error(
            "base sample must be the published pilot sample with the same seed",
            lambda malformed=malformed: runner._validate_base_identity(malformed, seed=42),
        )


def test_frozen_selection_seed_requires_an_integer_metadata_value() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    for seed in (42.0, True):
        selection = {**sample["selection"], "seed": seed}
        _assert_value_error(
            "frozen sample seed must be an integer",
            lambda selection=selection: runner._require_declared_seed(selection),
        )
    runner._require_declared_seed(sample["selection"])


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


def test_extension_provenance_mode_requires_pinned_pool_and_historical_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    source = sample["source"]
    reconstructed_hash = "a" * 64

    runner._validate_base_source(
        sample,
        source["sha256"],
        reconstructed_hash,
        allow_reconstructed_pipeline_manifest=True,
    )

    _assert_value_error(
        "base sample and pipeline manifest must have the same SHA-256",
        lambda: runner._validate_base_source(sample, source["sha256"], reconstructed_hash),
    )
    monkeypatch.setattr(runner, "EXPECTED_LABELLED_POOL_SHA256", "0" * 64)
    _assert_value_error(
        "reconstructed pipeline manifest extension requires the pinned labelled pool SHA-256",
        lambda: runner._validate_base_source(
            sample,
            source["sha256"],
            reconstructed_hash,
            allow_reconstructed_pipeline_manifest=True,
        ),
    )

    altered = json.loads(json.dumps(sample))
    altered["source"]["pipeline_manifest_sha256"] = "b" * 64
    _assert_value_error(
        "base sample does not record the pinned historical pipeline manifest SHA-256",
        lambda: runner._validate_base_source(
            altered,
            source["sha256"],
            reconstructed_hash,
            allow_reconstructed_pipeline_manifest=True,
        ),
    )


def test_reconstructed_base_identity_rejects_each_pool_hash_mismatch() -> None:
    source = {
        "pipeline_manifest_sha256": runner.EXPECTED_BASE_PIPELINE_MANIFEST_SHA256,
        "sha256": runner.EXPECTED_LABELLED_POOL_SHA256,
    }

    runner._validate_reconstructed_base_identity(source, runner.EXPECTED_LABELLED_POOL_SHA256)
    _assert_value_error(
        "reconstructed pipeline manifest extension requires the pinned labelled pool SHA-256",
        lambda: runner._validate_reconstructed_base_identity(source, "0" * 64),
    )
    _assert_value_error(
        "reconstructed pipeline manifest extension requires the pinned labelled pool SHA-256",
        lambda: runner._validate_reconstructed_base_identity(
            {**source, "sha256": "0" * 64}, runner.EXPECTED_LABELLED_POOL_SHA256
        ),
    )


def test_reconstructed_manifest_provenance_builder_rejects_unpinned_inputs() -> None:
    _assert_value_error(
        "reconstructed pipeline manifest extension requires the pinned labelled pool SHA-256",
        lambda: runner._reconstructed_manifest_provenance("a" * 64, "0" * 64),
    )
    _assert_value_error(
        "reconstructed pipeline manifest must differ from the historical manifest",
        lambda: runner._reconstructed_manifest_provenance(
            runner.EXPECTED_BASE_PIPELINE_MANIFEST_SHA256,
            runner.EXPECTED_LABELLED_POOL_SHA256,
        ),
    )


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ("not-json", "base frozen manifest.json is unreadable"),
        ("[]", "base frozen manifest.json must contain a JSON object"),
    ],
)
def test_base_manifest_reader_reports_exact_parse_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    error: str,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_BASE_MANIFEST_SHA256", runner.sha256_file(path))

    _assert_value_error(error, lambda: runner._read_base_manifest(path))


def test_base_manifest_reader_requires_the_published_checksum(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")

    _assert_value_error(
        "base frozen manifest checksum does not match the published pilot",
        lambda: runner._read_base_manifest(path),
    )


def test_base_manifest_reader_uses_explicit_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_BASE_MANIFEST_SHA256", runner.sha256_file(path))
    original_read_text = Path.read_text
    encodings: list[str | None] = []

    def record_encoding(path: Path, *args: Any, **kwargs: Any) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_encoding)

    assert runner._read_base_manifest(path) == {}
    assert encodings == ["utf-8"]


def test_base_manifest_output_validation_checks_each_pinned_output() -> None:
    sample_hash = "a" * 64
    _assert_value_error(
        "base frozen manifest is missing input or output checksums",
        lambda: runner._validate_base_manifest_outputs(None, sample_hash),
    )
    _assert_value_error(
        "base frozen manifest checksums do not match the published sample",
        lambda: runner._validate_base_manifest_outputs({}, sample_hash),
    )
    _assert_value_error(
        "base frozen manifest checksums do not match the published sample",
        lambda: runner._validate_base_manifest_outputs(
            {"frozen_sample.json": sample_hash}, sample_hash
        ),
    )
    _assert_value_error(
        "base frozen manifest checksums do not match the published sample",
        lambda: runner._validate_base_manifest_outputs(
            {"frozen_sample.json": sample_hash, "candidate_labels.csv": "0" * 64},
            sample_hash,
        ),
    )


def test_base_manifest_source_validation_checks_both_pinned_inputs() -> None:
    source = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))["source"]
    _assert_value_error(
        "base frozen manifest is missing input or output checksums",
        lambda: runner._validate_base_manifest_sources(None, source),
    )
    _assert_value_error(
        "base frozen manifest checksums do not match the published sample",
        lambda: runner._validate_base_manifest_sources({}, source),
    )
    _assert_value_error(
        "base frozen manifest checksums do not match the published sample",
        lambda: runner._validate_base_manifest_sources(
            {runner.LABELLED_NAME: source["sha256"]}, source
        ),
    )
    _assert_value_error(
        "base frozen manifest checksums do not match the published sample",
        lambda: runner._validate_base_manifest_sources(
            {
                runner.LABELLED_NAME: source["sha256"],
                runner.PIPELINE_MANIFEST_NAME: "0" * 64,
            },
            source,
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "other-method"),
        ("seed", 43),
        ("per_group", 49),
        ("sample_size", 98),
        ("sample_ids_sha256", "0" * 64),
    ],
)
def test_base_manifest_selection_validation_checks_every_protocol_field(
    field: str, value: Any
) -> None:
    selection = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))["selection"]
    manifest = {
        key: selection[key]
        for key in ("method", "seed", "per_group", "sample_size", "sample_ids_sha256")
    }
    manifest[field] = value

    _assert_value_error(
        "base frozen manifest selection does not match the published sample",
        lambda: runner._validate_base_manifest_selection(manifest, selection),
    )


def test_published_base_prefix_accepts_exactly_the_frozen_100_rows() -> None:
    rows = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))["selected_rows"]

    runner._require_published_base_prefix(rows)


def test_base_manifest_sample_checksum_error_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "frozen_sample.json"
    path.write_text("{}", encoding="utf-8")

    _assert_value_error(
        "base frozen sample checksum does not match the published pilot",
        lambda: runner._validate_base_manifest(path, {}),
    )


def test_default_row_validation_recomputes_base_centres_but_pinned_base_does_not() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    rows = [dict(row) for row in sample["selected_rows"]]
    rows[49]["cell_centre_lon"] = math.nextafter(rows[49]["cell_centre_lon"], math.inf)

    _assert_value_error(
        "frozen sample H3 cell does not match its centre",
        lambda: runner._validate_frozen_rows(rows, sample["selection"]),
    )
    _assert_value_error(
        "frozen sample H3 cell does not match its centre",
        lambda: runner._validate_frozen_selection(rows, sample["selection"]),
    )
    runner._validate_frozen_rows(rows, sample["selection"], pinned_base=True)


def test_row_hashing_uses_the_explicit_utf8_protocol() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    row = dict(sample["selected_rows"][0])
    encodings: list[str] = []

    class EncodingRecordingSentence(str):
        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            encodings.append(encoding)
            return super().encode(encoding, errors)

    row["sentence"] = EncodingRecordingSentence(row["sentence"])

    runner._validate_frozen_row_content(row)

    assert encodings == ["utf-8"]


def test_reconstructed_manifest_mode_requires_a_base_sample(tmp_path: Path) -> None:
    _assert_value_error(
        "reconstructed pipeline manifest mode requires --extend-from",
        lambda: runner.freeze_sample(
            tmp_path / "missing-pool.parquet",
            _candidate_file(),
            tmp_path / "missing-manifest.json",
            tmp_path / "new-run",
            allow_reconstructed_pipeline_manifest=True,
        ),
    )


def test_reconstructed_manifest_provenance_is_explicit_and_validated() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    source = dict(sample["source"])
    reconstructed_hash = "a" * 64
    provenance = runner._reconstructed_manifest_provenance(reconstructed_hash, source["sha256"])
    source["pipeline_manifest_sha256"] = reconstructed_hash
    source["pipeline_manifest_provenance"] = provenance

    runner._validate_frozen_source(source)

    assert provenance == {
        "mode": "reconstructed_manifest_same_labelled_pool",
        "historical_pipeline_manifest_sha256": runner.EXPECTED_BASE_PIPELINE_MANIFEST_SHA256,
        "extension_pipeline_manifest_sha256": reconstructed_hash,
        "labelled_pool_sha256": source["sha256"],
        "base_frozen_sample_sha256": runner.EXPECTED_BASE_FROZEN_SAMPLE_SHA256,
        "base_frozen_manifest_sha256": runner.EXPECTED_BASE_MANIFEST_SHA256,
        "base_selected_rows_sha256": runner.EXPECTED_BASE_SELECTED_ROWS_SHA256,
    }

    malformed = {**source, "pipeline_manifest_provenance": {**provenance}}
    malformed["pipeline_manifest_provenance"]["historical_pipeline_manifest_sha256"] = "b" * 64
    _assert_value_error(
        "frozen sample reconstructed-manifest provenance is inconsistent",
        lambda: runner._validate_frozen_source(malformed),
    )

    for bad_provenance in (None, []):
        malformed_type = {**source, "pipeline_manifest_provenance": bad_provenance}
        _assert_value_error(
            "frozen sample reconstructed-manifest provenance is inconsistent",
            lambda malformed_type=malformed_type: runner._validate_frozen_source(malformed_type),
        )

    bad_pool = {**source, "sha256": "0" * 64}
    bad_pool["pipeline_manifest_provenance"] = {
        **provenance,
        "labelled_pool_sha256": bad_pool["sha256"],
    }
    _assert_value_error(
        "frozen sample reconstructed-manifest provenance is inconsistent",
        lambda: runner._validate_frozen_source(bad_pool),
    )

    historical_manifest = {
        **source,
        "pipeline_manifest_sha256": runner.EXPECTED_BASE_PIPELINE_MANIFEST_SHA256,
    }
    historical_manifest["pipeline_manifest_provenance"] = {
        **provenance,
        "extension_pipeline_manifest_sha256": runner.EXPECTED_BASE_PIPELINE_MANIFEST_SHA256,
    }
    _assert_value_error(
        "frozen sample reconstructed-manifest provenance is inconsistent",
        lambda: runner._validate_frozen_source(historical_manifest),
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

    for field, value, message in (
        ("sample_size", 100.0, "DSpark evaluation requires a supported pilot sample size"),
        ("sample_size", True, "DSpark evaluation requires a supported pilot sample size"),
        ("seed", 42.0, "frozen sample seed does not match the published pilot"),
        ("seed", True, "frozen sample seed does not match the published pilot"),
    ):
        selection = {"sample_size": 100, "seed": 42}
        selection[field] = value
        _assert_value_error(
            message,
            lambda selection=selection: dspark_runner._validate_sample_selection(
                selection, [{}] * 100
            ),
        )


def test_frozen_manifest_requires_integer_seed_and_sample_size() -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    valid_manifest = {
        key: sample["selection"][key] for key in ("sample_ids_sha256", "seed", "sample_size")
    }
    dspark_runner._verify_frozen_selection(valid_manifest, sample)

    for field, value in (("seed", 42.0), ("sample_size", 100.0)):
        malformed = {**valid_manifest, field: value}
        _assert_value_error(
            "frozen manifest does not match frozen_sample.json",
            lambda malformed=malformed: dspark_runner._verify_frozen_selection(malformed, sample),
        )


def test_select_frozen_rows_forwards_seed_to_geographic_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = json.loads(BASE_SAMPLE.read_text(encoding="utf-8"))
    calls: list[int] = []

    def record_seed(*_args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        calls.append(kwargs["seed"])
        return [], {"sentinel": True}

    monkeypatch.setattr(runner, "extend_region_balanced_sample", record_seed)
    monkeypatch.setattr(
        runner, "_extension_geographic_metadata", lambda _rows, _metadata: {"ok": True}
    )
    rows, method, geography = runner._select_frozen_rows(
        [],
        500,
        42,
        BASE_SAMPLE,
        sample["source"]["sha256"],
        sample["source"]["pipeline_manifest_sha256"],
        allow_reconstructed_pipeline_manifest=False,
    )
    assert rows == sample["selected_rows"]
    assert method == runner.EXTENDED_SELECTION_METHOD
    assert geography == {"ok": True}
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
