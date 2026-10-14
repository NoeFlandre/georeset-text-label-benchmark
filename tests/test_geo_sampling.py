"""Contract tests for the geographic yes/no EUNIS sample (pure functions, fake grid)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import pytest

from georeset_text_label_benchmark.errors import DataValidationError
from georeset_text_label_benchmark.pilot import geo_sampling
from georeset_text_label_benchmark.pilot.geo_sampling import (
    EARTH_RADIUS_KM,
    H3_RESOLUTION,
    _haversine_km,
    _occurrence_id,
    _region_flow_value,
    _stable_key,
    bbox_centre,
    extend_region_balanced_sample,
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


def _stratified_point_row(
    osm_id: int, decision: str, stratum: str, lat: float, lon: float
) -> dict[str, Any]:
    return {**_point_row(osm_id, decision, lat, lon), "m49_stratum": stratum}


def _stratified_existing_row(
    osm_id: int, decision: str, stratum: str, lat: float, lon: float
) -> dict[str, Any]:
    row = _stratified_point_row(osm_id, decision, stratum, lat, lon)
    cell = _point_cell_of(lat, lon)
    return {
        **row,
        "h3_cell": cell,
        "cell_centre_lat": lat,
        "cell_centre_lon": lon,
    }


def _stratified_extend(
    rows: list[dict[str, Any]], existing: list[dict[str, Any]], per_group: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return extend_region_balanced_sample(
        rows,
        cell_of=_point_cell_of,
        centre_of=_point_centre_of,
        stratum_of=lambda row: row["m49_stratum"],
        existing=existing,
        per_group=per_group,
        seed=SEED,
    )


def test_region_balanced_extension_preserves_prefix_and_uses_common_capacity_aware_quotas() -> None:
    existing = [
        _stratified_existing_row(1, "yes", "039", 0.0, 0.0),
        _stratified_existing_row(2, "yes", "154", 10.0, 0.0),
        _stratified_existing_row(3, "no", "154", 10.0, 10.0),
        _stratified_existing_row(4, "no", "UNASSIGNED", -10.0, 10.0),
    ]
    candidates = [
        _stratified_point_row(10, "yes", "039", 1.0, 0.0),
        _stratified_point_row(11, "yes", "154", 11.0, 0.0),
        _stratified_point_row(12, "yes", "151", 20.0, 0.0),
        _stratified_point_row(13, "yes", "151", 21.0, 0.0),
        _stratified_point_row(14, "yes", "UNASSIGNED", 30.0, 0.0),
        _stratified_point_row(15, "yes", "UNASSIGNED", 31.0, 0.0),
        _stratified_point_row(20, "no", "039", 1.0, 10.0),
        _stratified_point_row(21, "no", "154", 11.0, 10.0),
        _stratified_point_row(22, "no", "151", 20.0, 10.0),
        _stratified_point_row(23, "no", "151", 21.0, 10.0),
        _stratified_point_row(24, "no", "UNASSIGNED", 30.0, 10.0),
        _stratified_point_row(25, "no", "UNASSIGNED", 31.0, 10.0),
    ]

    added, metadata = _stratified_extend(candidates, existing, per_group=6)
    repeated, _ = _stratified_extend(list(reversed(candidates)), existing, per_group=6)
    selected = [*existing, *added]

    assert repeated == added
    assert selected[: len(existing)] == existing
    quotas = metadata["target_counts_per_group"]
    assert set(quotas) == {"039", "151", "154", "UNASSIGNED"}
    assert sorted(quotas.values()) == [1, 1, 2, 2]
    assert max(quotas.values()) - min(quotas.values()) == 1
    assert (
        metadata["selected_counts_by_decision"]["yes"]
        == metadata["selected_counts_by_decision"]["no"]
        == metadata["target_counts_per_group"]
    )
    for decision in ("yes", "no"):
        selected_rows = [row for row in selected if row["decision"] == decision]
        assert len(selected_rows) == 6
        assert len({row["h3_cell"] for row in selected_rows}) == 6
    assert len({row["text_sha256"] for row in selected}) == len(selected)
    assert len({(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in selected}) == len(
        selected
    )


def _flow_cell_of(lat: float, lon: float) -> str:
    group = "yes" if lon < 30 else "no"
    if lat < 1:
        return f"{group}-shared"
    return f"{group}-a" if lat < 15 else f"{group}-b"


def _flow_centre_of(cell: str) -> tuple[float, float]:
    group, location = cell.split("-")
    latitude = {"shared": 0.0, "a": 10.0, "b": 20.0}[location]
    longitude = 0.0 if group == "yes" else 60.0
    return latitude, longitude


def test_cell_region_flow_preserves_feasibility_when_a_cell_has_multiple_regions() -> None:
    candidates = [
        _stratified_point_row(10, "yes", "A", 0.0, 0.0),
        _stratified_point_row(11, "yes", "B", 0.2, 0.0),
        _stratified_point_row(12, "yes", "A", 10.0, 0.0),
        _stratified_point_row(20, "no", "A", 0.0, 60.0),
        _stratified_point_row(21, "no", "B", 0.2, 60.0),
        _stratified_point_row(22, "no", "A", 10.0, 60.0),
    ]

    added, metadata = extend_region_balanced_sample(
        candidates,
        cell_of=_flow_cell_of,
        centre_of=_flow_centre_of,
        stratum_of=lambda row: row["m49_stratum"],
        existing=[],
        per_group=2,
        seed=SEED,
    )

    assert metadata["target_counts_per_group"] == {"A": 1, "B": 1}
    assert metadata["candidate_multi_region_cell_count_by_decision"] == {"yes": 1, "no": 1}
    assert metadata["candidate_unique_cell_count_by_decision"] == {"yes": 2, "no": 2}
    for group in ("yes", "no"):
        rows = [row for row in added if row["decision"] == group]
        assert {row["m49_stratum"] for row in rows} == {"A", "B"}
        assert len({row["h3_cell"] for row in rows}) == 2


def test_cell_region_flow_does_not_count_a_multi_region_cell_twice() -> None:
    candidates = [
        _stratified_point_row(10, "yes", "A", 0.0, 0.0),
        _stratified_point_row(11, "yes", "B", 0.2, 0.0),
        _stratified_point_row(20, "no", "A", 10.0, 60.0),
        _stratified_point_row(21, "no", "B", 20.0, 60.0),
    ]

    with pytest.raises(ValueError, match="cannot meet common regional quotas") as error:
        extend_region_balanced_sample(
            candidates,
            cell_of=_flow_cell_of,
            centre_of=_flow_centre_of,
            stratum_of=lambda row: row["m49_stratum"],
            existing=[],
            per_group=2,
            seed=SEED,
        )

    assert '"additional_flow_capacity_by_decision": {"no": 2, "yes": 1}' in str(error.value)


def test_region_flow_value_reports_bottlenecked_capacity_when_cells_are_fewer_than_demand() -> None:
    cells = {
        **{f"a-{index}": {"A": []} for index in range(4)},
        "b-0": {"B": []},
    }

    assert _region_flow_value(cells, {"A": 3, "B": 3}) == 4


def test_global_cell_capacity_rejects_shortfall_before_quota_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _stratified_point_row(
            index * 10 + offset,
            decision,
            f"region-{index}",
            float(index * 10),
            0.0 if decision == "yes" else 60.0,
        )
        for index in range(8)
        for offset, decision in enumerate(("yes", "no"))
    ]

    def cell_of(_latitude: float, longitude: float) -> str:
        return "yes-shared" if longitude < 30 else "no-shared"

    def search_must_not_run(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("quota search must stop when global candidate cells cannot fill the target")

    monkeypatch.setattr(geo_sampling, "_search_common_quota_completion", search_must_not_run)
    with pytest.raises(ValueError, match="cannot meet common regional quotas") as error:
        extend_region_balanced_sample(
            rows,
            cell_of=cell_of,
            centre_of=lambda _cell: (0.0, 0.0),
            stratum_of=lambda row: row["m49_stratum"],
            existing=[],
            per_group=2,
            seed=SEED,
        )

    assert '"additional_flow_capacity_by_decision": {"no": 1, "yes": 1}' in str(error.value)


def test_common_quota_selection_backtracks_past_a_feasible_dead_end() -> None:
    candidates = [
        _stratified_point_row(10, "yes", "C", 0.0, 0.0),
        _stratified_point_row(11, "yes", "A", 10.0, 0.0),
        _stratified_point_row(12, "yes", "B", 10.0, 0.0),
        _stratified_point_row(20, "no", "B", 0.0, 60.0),
        _stratified_point_row(21, "no", "A", 10.0, 60.0),
        _stratified_point_row(22, "no", "C", 10.0, 60.0),
    ]

    def cell_of(latitude: float, longitude: float) -> str:
        if longitude < 30.0:
            return "yes-c" if latitude < 1.0 else "yes-ab"
        return "no-b" if latitude < 1.0 else "no-ac"

    centres = {
        "yes-c": (0.0, 0.0),
        "yes-ab": (10.0, 0.0),
        "no-b": (0.0, 60.0),
        "no-ac": (10.0, 60.0),
    }
    added, metadata = extend_region_balanced_sample(
        candidates,
        cell_of=cell_of,
        centre_of=centres.__getitem__,
        stratum_of=lambda row: row["m49_stratum"],
        existing=[],
        per_group=2,
        seed=SEED,
    )

    assert metadata["target_counts_per_group"] == {"A": 0, "B": 1, "C": 1}
    for group in ("yes", "no"):
        assert {row["m49_stratum"] for row in added if row["decision"] == group} == {"B", "C"}


def test_region_balanced_extension_reports_an_unmatchable_prefix_and_capacity_shortfall() -> None:
    existing = [
        _stratified_existing_row(1, "yes", "039", 0.0, 0.0),
        _stratified_existing_row(2, "no", "154", 10.0, 10.0),
    ]
    candidates = [
        _stratified_point_row(3, "yes", "039", 1.0, 0.0),
        _stratified_point_row(4, "yes", "039", 2.0, 0.0),
        _stratified_point_row(5, "no", "154", 11.0, 10.0),
        _stratified_point_row(6, "no", "154", 12.0, 10.0),
    ]

    with pytest.raises(ValueError, match="cannot meet common regional quotas") as error:
        _stratified_extend(candidates, existing, per_group=3)

    assert '"shortfall_per_group": 1' in str(error.value)
    assert '"unmatchable_prefix_strata": {"039": [1, 0], "154": [1, 0]}' in str(error.value)


def test_region_maximin_spacing_uses_only_prefix_anchors_in_the_same_stratum() -> None:
    existing = [
        _stratified_existing_row(1, "yes", "A", 0.0, 0.0),
        _stratified_existing_row(2, "yes", "B", 0.0, 91.0),
        _stratified_existing_row(3, "no", "A", 0.0, 30.0),
        _stratified_existing_row(4, "no", "B", 0.0, 92.0),
    ]
    candidates = [
        _stratified_point_row(10, "yes", "A", 0.0, 10.0),
        _stratified_point_row(11, "yes", "A", 0.0, 90.0),
        _stratified_point_row(12, "yes", "B", 20.0, 20.0),
        _stratified_point_row(13, "yes", "B", 30.0, 30.0),
        _stratified_point_row(20, "no", "A", 0.0, 40.0),
        _stratified_point_row(21, "no", "A", 0.0, 50.0),
        _stratified_point_row(22, "no", "B", 40.0, 40.0),
        _stratified_point_row(23, "no", "B", 50.0, 50.0),
    ]

    added, _ = _stratified_extend(candidates, existing, per_group=4)
    yes_a = [row for row in added if row["decision"] == "yes" and row["m49_stratum"] == "A"]

    assert len(yes_a) == 1
    assert yes_a[0]["osm_id"] == 11


def test_incremental_region_distances_preserve_seeded_candidate_order() -> None:
    cells = {
        "a0": {"A": []},
        "a1": {"A": []},
        "ab": {"A": [], "B": []},
        "b0": {"B": []},
        "b1": {"B": []},
        "b2": {"B": []},
    }
    centres = {
        "a0": (0.0, 0.0),
        "a1": (0.0, 10.0),
        "ab": (20.0, 0.0),
        "b0": (10.0, 10.0),
        "b1": (5.0, 10.0),
        "b2": (0.0, 10.0),
        "anchor-a": (1.0, 1.0),
        "anchor-b": (9.0, 9.0),
    }
    anchors = {"A": [centres["anchor-a"]], "B": [centres["anchor-b"]]}
    nearest = geo_sampling._nearest_region_distances(cells, centres, anchors)
    selected_cells: set[str] = set()

    def assert_ranking_matches_reference(stratum: str) -> list[str]:
        expected = sorted(
            (
                cell
                for cell, regions in cells.items()
                if stratum in regions and cell not in selected_cells
            ),
            key=lambda cell: (
                -geo_sampling._nearest_anchor_distance(centres[cell], anchors[stratum]),
                _stable_key(SEED, cell),
            ),
        )
        actual = geo_sampling._rank_maximin_candidates(
            stratum, cells, selected_cells, nearest, SEED
        )
        assert actual == expected
        return actual

    first_a_ranking = assert_ranking_matches_reference("A")
    assert first_a_ranking[0] == "ab"
    selected_cells.add("ab")
    geo_sampling._update_nearest_region_distances("A", cells, centres, nearest, "ab")
    anchors["A"].append(centres["ab"])
    assert_ranking_matches_reference("A")

    first_b_ranking = assert_ranking_matches_reference("B")
    assert "ab" not in first_b_ranking
    selected = first_b_ranking[0]
    selected_cells.add(selected)
    geo_sampling._update_nearest_region_distances("B", cells, centres, nearest, selected)
    anchors["B"].append(centres[selected])
    assert_ranking_matches_reference("B")


def test_region_balanced_extension_requires_a_nonempty_stratum() -> None:
    base = _stratified_existing_row(1, "yes", "039", 0.0, 0.0)
    no_base = _stratified_existing_row(2, "no", "039", 0.0, 10.0)
    candidates = [
        _stratified_point_row(3, "yes", "", 1.0, 0.0),
        _stratified_point_row(4, "no", "039", 1.0, 10.0),
    ]

    with pytest.raises(ValueError, match="geographic stratum must be a non-empty string"):
        _stratified_extend(candidates, [base, no_base], per_group=2)


def test_region_extension_skips_candidate_cells_already_in_the_prefix() -> None:
    existing = [
        _stratified_existing_row(1, "yes", "A", 0.0, 0.0),
        _stratified_existing_row(2, "no", "A", 0.0, 10.0),
    ]
    candidates = [
        _stratified_point_row(3, "yes", "A", 0.0, 0.0),
        _stratified_point_row(4, "yes", "A", 0.0, 1.0),
        _stratified_point_row(5, "no", "A", 0.0, 10.0),
        _stratified_point_row(6, "no", "A", 0.0, 11.0),
    ]

    added, _ = _stratified_extend(candidates, existing, per_group=2)

    assert {row["osm_id"] for row in added} == {4, 6}


def test_region_extension_reports_a_prefix_that_cannot_fit_the_requested_size() -> None:
    existing = [
        _stratified_existing_row(1, "yes", "A", 0.0, 0.0),
        _stratified_existing_row(2, "yes", "A", 1.0, 0.0),
        _stratified_existing_row(3, "yes", "B", 2.0, 0.0),
        _stratified_existing_row(4, "no", "A", 0.0, 10.0),
        _stratified_existing_row(5, "no", "B", 1.0, 10.0),
        _stratified_existing_row(6, "no", "B", 2.0, 10.0),
    ]

    with pytest.raises(ValueError, match="cannot meet common regional quotas") as error:
        _stratified_extend([], existing, per_group=3)

    assert '"allocated_common_total": 4' in str(error.value)
    assert '"shortfall_per_group": 0' in str(error.value)


def test_region_extension_reports_no_shared_candidate_capacity() -> None:
    existing = [
        _stratified_existing_row(1, "yes", "A", 0.0, 0.0),
        _stratified_existing_row(2, "no", "A", 0.0, 10.0),
    ]

    with pytest.raises(ValueError, match="cannot meet common regional quotas") as error:
        _stratified_extend([], existing, per_group=2)

    assert '"shortfall_per_group": 1' in str(error.value)
    assert '"allocated_common_total": 1' in str(error.value)


def test_region_extension_rejects_nonpositive_size_before_reading_rows() -> None:
    with pytest.raises(ValueError, match="per_group must be positive"):
        _stratified_extend([], [], per_group=0)


def test_stratification_helpers_fail_closed_on_inconsistent_internal_capacity() -> None:
    with pytest.raises(ValueError, match="exhausted feasible A cells"):
        geo_sampling._maximin_feasible_cell("A", {}, {"A": 1}, set(), {}, SEED)
    with pytest.raises(ValueError, match="did not fill the requested sample size"):
        geo_sampling._require_extension_fill([], [], per_group=1)


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
