"""Contract tests for the geographic yes/no EUNIS sample (pure functions, fake grid)."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pytest

from georeset_text_label_benchmark.errors import DataValidationError
from georeset_text_label_benchmark.pilot.geo_sampling import (
    H3_RESOLUTION,
    bbox_centre,
    h3_cell_of,
    h3_centre_of,
    select_geographic_sample,
)


def _cell_of(lat: float, lon: float) -> str:
    return f"{math.floor(lat / 10)}:{math.floor(lon / 10)}"


def _centre_of(cell: str) -> tuple[float, float]:
    row, column = (int(part) for part in cell.split(":"))
    return row * 10 + 5.0, column * 10 + 5.0


def _row(
    osm_id: int,
    decision: str,
    lat: float,
    lon: float,
    *,
    text: str | None = None,
    language: str | None = "eng",
    code: str | None = "T11",
) -> dict[str, Any]:
    sentence = text if text is not None else f"Sentence {osm_id} {decision}."
    return {
        "source_pbf": "tuvalu-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": osm_id,
        "description_identity": f"desc-{osm_id}",
        "tag_key": "description",
        "sentence_index": 0,
        "sentence": sentence,
        "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
        "language_code": language,
        "eunis_code": code,
        "eunis_name": "Habitat" if code else None,
        "eunis_overlap_percentage": 80.0 if code else None,
        "eunis_source_version": "maps-v1" if code else None,
        "decision": decision,
        "bbox_min_x": lon - 0.1,
        "bbox_min_y": lat - 0.1,
        "bbox_max_x": lon + 0.1,
        "bbox_max_y": lat + 0.1,
    }


def _select(rows: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    return select_geographic_sample(
        rows, cell_of=_cell_of, centre_of=_centre_of, **{"per_group": 3, "seed": 42, **kwargs}
    )


def _grid_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    osm_id = 0
    for decision in ("yes", "no"):
        for cell_row in range(-4, 4):
            for cell_column in range(-4, 4):
                osm_id += 1
                lat = cell_row * 10 + 5.0
                lon = cell_column * 10 + 5.0
                rows.append(_row(osm_id, decision, lat, lon))
    return rows


def test_bbox_centre_is_the_midpoint_returned_as_latitude_longitude() -> None:
    row = {"bbox_min_x": 10.0, "bbox_min_y": 20.0, "bbox_max_x": 14.0, "bbox_max_y": 26.0}

    assert bbox_centre(row) == (23.0, 12.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bbox_min_x", float("nan")),
        ("bbox_max_y", float("inf")),
        ("bbox_min_y", "20"),
        ("bbox_max_x", True),
        ("bbox_min_x", 200.0),
    ],
)
def test_bbox_centre_rejects_non_finite_or_out_of_range_values(field: str, value: Any) -> None:
    row = {"bbox_min_x": 10.0, "bbox_min_y": 20.0, "bbox_max_x": 14.0, "bbox_max_y": 26.0}
    row[field] = value

    with pytest.raises(DataValidationError, match=field):
        bbox_centre(row)


def test_selects_equal_yes_and_no_sentences_with_one_per_cell() -> None:
    sample = _select(_grid_rows(), per_group=3)

    decisions = [row["decision"] for row in sample]
    assert decisions.count("yes") == 3
    assert decisions.count("no") == 3
    cells = [row["h3_cell"] for row in sample]
    assert len(set(cells)) == len(cells) == 6


def test_yes_and_no_never_share_a_cell() -> None:
    sample = _select(_grid_rows(), per_group=8)

    yes_cells = {row["h3_cell"] for row in sample if row["decision"] == "yes"}
    no_cells = {row["h3_cell"] for row in sample if row["decision"] == "no"}
    assert yes_cells.isdisjoint(no_cells)


def test_a_far_cluster_is_always_chosen_over_a_second_near_cell() -> None:
    near_a = [_row(1, "yes", 5.0, 5.0)]
    near_b = [_row(2, "yes", 5.0, 15.0)]
    far = [_row(3, "yes", 65.0, 125.0)]
    no_rows = [_row(4, "no", -55.0, -175.0), _row(5, "no", -55.0, -165.0)]

    sample = _select(near_a + near_b + far + no_rows, per_group=2)

    yes_cells = {row["h3_cell"] for row in sample if row["decision"] == "yes"}
    assert "6:12" in yes_cells


def test_same_input_and_seed_give_the_same_sample() -> None:
    first = _select(_grid_rows(), per_group=4, seed=7)
    second = _select(list(reversed(_grid_rows())), per_group=4, seed=7)

    assert [row["osm_id"] for row in first] == [row["osm_id"] for row in second]


def test_only_english_rows_with_an_eunis_assignment_are_eligible() -> None:
    rows = [
        _row(1, "yes", 5.0, 5.0),
        _row(2, "yes", 15.0, 15.0, language="fra"),
        _row(3, "yes", 25.0, 25.0, language=None),
        _row(4, "yes", 35.0, 35.0, code=None),
        _row(5, "no", 45.0, 45.0),
    ]

    sample = _select(rows, per_group=1)

    assert {row["osm_id"] for row in sample} == {1, 5}


def test_duplicate_sentence_text_is_used_at_most_once() -> None:
    rows = [
        _row(1, "yes", 5.0, 5.0, text="The same sentence."),
        _row(2, "yes", 55.0, 55.0, text="The same sentence."),
        _row(3, "no", -45.0, -45.0),
    ]

    sample = _select(rows, per_group=1)

    texts = [row["sentence"] for row in sample]
    assert len(texts) == len(set(texts))


def test_duplicate_polygon_is_used_at_most_once() -> None:
    first = _row(1, "yes", 5.0, 5.0)
    second = {**_row(2, "yes", 55.0, 55.0), "osm_id": 1, "source_pbf": first["source_pbf"]}
    rows = [first, second, _row(3, "no", -45.0, -45.0)]

    sample = _select(rows, per_group=1)

    polygons = [(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in sample]
    assert len(polygons) == len(set(polygons))


def test_insufficient_cells_for_a_group_fail_with_the_group_named() -> None:
    rows = [_row(1, "yes", 5.0, 5.0), _row(2, "no", 15.0, 15.0), _row(3, "no", 25.0, 25.0)]

    with pytest.raises(ValueError, match="need 2 yes cells, found 1"):
        _select(rows, per_group=2)


def test_unknown_decision_on_an_assigned_row_fails() -> None:
    rows = [_row(1, "maybe", 5.0, 5.0)]

    with pytest.raises(ValueError, match="unknown decision"):
        _select(rows, per_group=1)


def test_default_grid_uses_h3_resolution_three_cells() -> None:
    cell = h3_cell_of(48.85, 2.35)

    assert isinstance(cell, str)
    assert H3_RESOLUTION == 3
    assert __import__("h3").get_resolution(cell) == 3
    lat, lon = h3_centre_of(cell)
    assert abs(lat - 48.85) < 3.0
    assert abs(lon - 2.35) < 3.0
