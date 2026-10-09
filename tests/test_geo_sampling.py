"""Contract tests for the geographic yes/no EUNIS sample (pure functions, fake grid)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import pytest

from georeset_text_label_benchmark.errors import DataValidationError
from georeset_text_label_benchmark.pilot.geo_sampling import (
    EARTH_RADIUS_KM,
    H3_RESOLUTION,
    _haversine_km,
    _occurrence_id,
    _stable_key,
    bbox_centre,
    h3_cell_of,
    h3_centre_of,
    select_geographic_sample,
)

SEED = 42
BOX = {"bbox_min_x": 10.0, "bbox_min_y": 20.0, "bbox_max_x": 14.0, "bbox_max_y": 26.0}
YES_POINTS = [(0.0, 0.0), (0.0, 30.0), (20.0, 60.0), (-30.0, -30.0), (45.0, 90.0), (-60.0, 120.0)]
NO_POINTS = [
    (5.0, 5.0),
    (10.0, -80.0),
    (-20.0, 150.0),
    (60.0, -40.0),
    (35.0, 35.0),
    (-45.0, -100.0),
]


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

    with pytest.raises(ValueError, match=_exact("unknown decision 'maybe'")):
        _select(rows, per_group=1)


def test_default_grid_uses_h3_resolution_three_cells() -> None:
    cell = h3_cell_of(48.85, 2.35)

    assert isinstance(cell, str)
    assert H3_RESOLUTION == 3
    assert __import__("h3").get_resolution(cell) == 3
    lat, lon = h3_centre_of(cell)
    assert abs(lat - 48.85) < 3.0
    assert abs(lon - 2.35) < 3.0


def test_groups_that_cannot_take_disjoint_cells_fail_closed() -> None:
    rows = [_row(1, "yes", 5.0, 5.0), _row(2, "no", 5.0, 5.0)]

    with pytest.raises(ValueError, match="cannot choose 1 disjoint cells for yes"):
        _select(rows, per_group=1)


def test_per_group_must_be_positive() -> None:
    with pytest.raises(ValueError, match=_exact("per_group must be positive")):
        _select(_grid_rows(), per_group=0)


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


def _point_cell_of(lat: float, lon: float) -> str:
    return f"{lat!r}:{lon!r}"


def _point_centre_of(cell: str) -> tuple[float, float]:
    lat, lon = cell.split(":")
    return float(lat), float(lon)


def _point_row(
    osm_id: int, decision: str, lat: float, lon: float, *, text: str | None = None
) -> dict[str, Any]:
    """A row whose bbox is a single point, so its centre is exact."""
    return {
        **_row(osm_id, decision, lat, lon, text=text),
        "bbox_min_x": lon,
        "bbox_max_x": lon,
        "bbox_min_y": lat,
        "bbox_max_y": lat,
    }


def _point_rows(
    yes_points: list[tuple[float, float]], no_points: list[tuple[float, float]]
) -> list[dict[str, Any]]:
    yes = [_point_row(index, "yes", lat, lon) for index, (lat, lon) in enumerate(yes_points, 1)]
    no = [_point_row(100 + index, "no", lat, lon) for index, (lat, lon) in enumerate(no_points, 1)]
    return yes + no


def _select_points(rows: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    return select_geographic_sample(
        rows, cell_of=_point_cell_of, centre_of=_point_centre_of, **kwargs
    )


def _expected_pick(candidates: list[str], chosen: list[str]) -> str:
    """Restate the rule: first pick by seeded key, then farthest from every chosen cell."""
    if not chosen:
        return min(candidates, key=lambda cell: _stable_key(SEED, cell))
    chosen_centres = [_point_centre_of(cell) for cell in chosen]
    return max(
        candidates,
        key=lambda cell: (
            min(_haversine_km(_point_centre_of(cell), anchor) for anchor in chosen_centres),
            _stable_key(SEED, cell),
        ),
    )


def _assert_maximin_order(
    sample: list[dict[str, Any]], yes_cells: list[str], no_cells: list[str]
) -> None:
    chosen: list[str] = []
    for row in sample:
        pool = yes_cells if row["decision"] == "yes" else no_cells
        candidates = [cell for cell in pool if cell not in chosen]
        assert row["h3_cell"] == _expected_pick(candidates, chosen)
        chosen.append(row["h3_cell"])


@pytest.mark.parametrize(
    ("field", "limit"),
    [("bbox_min_x", 180.0), ("bbox_max_x", 180.0), ("bbox_min_y", 90.0), ("bbox_max_y", 90.0)],
)
def test_bbox_coordinates_accept_the_limit_and_reject_just_past_it(
    field: str, limit: float
) -> None:
    bbox_centre({**BOX, field: limit})
    bbox_centre({**BOX, field: -limit})

    with pytest.raises(DataValidationError) as error:
        bbox_centre({**BOX, field: limit + 0.5})

    assert str(error.value) == f"polygon {field} must be a finite coordinate within ±{limit:g}"


def test_default_group_size_is_fifty() -> None:
    yes = [_point_row(index, "yes", float(index - 25), 0.0) for index in range(50)]
    no = [_point_row(100 + index, "no", float(index - 25), 40.0) for index in range(50)]

    sample = select_geographic_sample(yes + no, cell_of=_point_cell_of, centre_of=_point_centre_of)

    assert len(sample) == 100


def test_each_pick_is_the_seeded_first_then_the_farthest_cell_from_every_chosen_cell() -> None:
    sample = _select_points(_point_rows(YES_POINTS, NO_POINTS), per_group=3)

    assert len(sample) == 6
    _assert_maximin_order(
        sample,
        [_point_cell_of(lat, lon) for lat, lon in YES_POINTS],
        [_point_cell_of(lat, lon) for lat, lon in NO_POINTS],
    )


def test_a_distance_tie_is_broken_by_the_seeded_key() -> None:
    rows = _point_rows([(0.0, 0.0)], [(0.0, 10.0), (0.0, -10.0), (10.0, 0.0), (-10.0, 0.0)])

    sample = _select_points(rows, per_group=1)

    no_cells = [
        _point_cell_of(0.0, 10.0),
        _point_cell_of(0.0, -10.0),
        _point_cell_of(10.0, 0.0),
        _point_cell_of(-10.0, 0.0),
    ]
    assert sample[1]["h3_cell"] == max(no_cells, key=lambda cell: _stable_key(SEED, cell))


def test_a_cell_taken_by_one_group_is_never_offered_again() -> None:
    rows = [
        _point_row(1, "yes", 0.0, 0.0),
        _point_row(2, "yes", 0.0, 10.0),
        _point_row(3, "no", 0.0, 0.0),
        _point_row(4, "no", 0.0, 20.0),
    ]

    with pytest.raises(ValueError, match=_exact("cannot choose 2 disjoint cells for yes")):
        _select_points(rows, per_group=2)


def test_text_shared_by_two_rows_leaves_one_candidate() -> None:
    rows = [
        _point_row(1, "yes", 10.0, 10.0, text="The same sentence."),
        _point_row(2, "yes", 20.0, 20.0, text="The same sentence."),
        _point_row(3, "yes", 30.0, 30.0, text="Another sentence."),
        _point_row(4, "no", -10.0, -10.0),
        _point_row(5, "no", -20.0, -20.0),
        _point_row(6, "no", -30.0, -30.0),
    ]

    with pytest.raises(ValueError, match=_exact("need 3 yes cells, found 2")):
        _select_points(rows, per_group=3)


def test_polygon_shared_by_two_rows_leaves_one_candidate() -> None:
    rows = [
        _point_row(1, "yes", 10.0, 10.0),
        _point_row(1, "yes", 20.0, 20.0, text="A second sentence."),
        _point_row(3, "yes", 30.0, 30.0),
        _point_row(4, "no", -10.0, -10.0),
        _point_row(5, "no", -20.0, -20.0),
        _point_row(6, "no", -30.0, -30.0),
    ]

    with pytest.raises(ValueError, match=_exact("need 3 yes cells, found 2")):
        _select_points(rows, per_group=3)


def test_each_shared_sentence_is_represented_by_its_seeded_first_occurrence() -> None:
    rows: list[dict[str, Any]] = []
    expected_ids: set[int] = set()
    for index in range(10):
        lat = float(-40 + 8 * index)
        first = _point_row(2 * index + 1, "yes", lat, 10.0, text=f"Shared sentence {index}.")
        second = _point_row(2 * index + 2, "yes", lat, 70.0, text=f"Shared sentence {index}.")
        rows += [second, first]
        winner = min((first, second), key=lambda row: _stable_key(SEED, _occurrence_id(row)))
        expected_ids.add(winner["osm_id"])
    rows += [_point_row(100 + index, "no", float(-40 + 8 * index), -60.0) for index in range(10)]

    sample = _select_points(rows, per_group=10)

    assert {row["osm_id"] for row in sample if row["decision"] == "yes"} == expected_ids


def test_occurrence_identity_joins_its_fields_in_order() -> None:
    row = _row(7, "yes", 0.0, 0.0, text="Hello.")

    assert _occurrence_id(row) == (
        f"tuvalu-latest.osm.pbf|way|7|desc-7|description|0|{row['text_sha256']}"
    )


def test_sample_rows_carry_their_cell_and_its_centre_as_latitude_and_longitude() -> None:
    sample = _select_points(_point_rows(YES_POINTS, NO_POINTS), per_group=3)

    for row in sample:
        lat, lon = _point_centre_of(row["h3_cell"])
        assert row["cell_centre_lat"] == lat
        assert row["cell_centre_lon"] == lon


@pytest.mark.parametrize(
    ("first", "second", "expected_km"),
    [
        ((0.0, 0.0), (0.0, 90.0), EARTH_RADIUS_KM * math.pi / 2),
        ((10.0, 0.0), (40.0, 0.0), EARTH_RADIUS_KM * math.radians(30.0)),
        ((0.0, 10.0), (0.0, 40.0), EARTH_RADIUS_KM * math.radians(30.0)),
        ((30.0, 10.0), (30.0, 10.0), 0.0),
    ],
)
def test_haversine_matches_great_circle_arcs(
    first: tuple[float, float], second: tuple[float, float], expected_km: float
) -> None:
    assert _haversine_km(first, second) == pytest.approx(expected_km, rel=1e-12, abs=1e-9)
