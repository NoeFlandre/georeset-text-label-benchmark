"""Tests for the pinned UN M49 region assignment asset."""

from __future__ import annotations

from typing import cast

import pytest

from georeset_text_label_benchmark.errors import DataValidationError
from georeset_text_label_benchmark.pilot import m49
from georeset_text_label_benchmark.pilot.m49 import (
    M49_BOUNDARIES_SHA256,
    M49_STRATIFICATION_VERSION,
    M49_UNASSIGNED,
    m49_assignment_for_coordinate,
    m49_assignment_for_row,
    m49_boundary_provenance,
)


@pytest.mark.parametrize(
    ("latitude", "longitude", "code", "name"),
    [
        (48.8566, 2.3522, "155", "Western Europe"),
        (52.2297, 21.0122, "151", "Eastern Europe"),
        (60.1699, 24.9384, "154", "Northern Europe"),
        (41.9028, 12.4964, "039", "Southern Europe"),
        (36.8065, 10.1815, "015", "Northern Africa"),
        (39.9334, 32.8597, "145", "Western Asia"),
        (47.067270750000006, 52.05008365, "143", "Central Asia"),
        (51.1694, 71.4491, "143", "Central Asia"),
    ],
)
def test_coordinate_assignment_uses_official_m49_subregion_codes(
    latitude: float, longitude: float, code: str, name: str
) -> None:
    assignment = m49_assignment_for_coordinate(latitude, longitude)

    assert assignment.stratum == code
    assert assignment.subregion_code == code
    assert assignment.subregion_name == name
    assert assignment.status == "assigned"


@pytest.mark.parametrize(
    ("latitude", "longitude", "status"),
    [(0.0, 0.0, "outside_country_geometry")],
)
def test_unmatched_or_cross_region_boundary_coordinates_are_not_nearest_assigned(
    latitude: float, longitude: float, status: str
) -> None:
    assignment = m49_assignment_for_coordinate(latitude, longitude)

    assert assignment.stratum == M49_UNASSIGNED
    assert assignment.subregion_code is None
    assert assignment.subregion_name is None
    assert assignment.status == status


@pytest.mark.parametrize(
    "second_code", ["039", None], ids=["different-subregions", "missing-crosswalk"]
)
def test_overlapping_regions_are_left_unassigned(
    monkeypatch: pytest.MonkeyPatch, second_code: str | None
) -> None:
    ring = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0))
    known = m49._Polygon("015", "Northern Africa", (ring,), (0.0, 0.0, 2.0, 2.0))
    second = m49._Polygon(
        second_code, "Southern Europe" if second_code else None, (ring,), (0, 0, 2, 2)
    )
    monkeypatch.setattr(m49, "_load_boundaries", lambda: ((known, second), {(18, 36): (0, 1)}))

    assignment = m49_assignment_for_coordinate(1.0, 1.0)

    assert assignment.stratum == M49_UNASSIGNED
    assert assignment.status == "ambiguous_boundary"


def test_repeated_coordinate_assignments_reuse_the_same_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0))
    polygon = m49._Polygon("015", "Northern Africa", (ring,), (0.0, 0.0, 2.0, 2.0))
    loads = 0

    def load_boundaries() -> tuple[
        tuple[m49._Polygon, ...], dict[tuple[int, int], tuple[int, ...]]
    ]:
        nonlocal loads
        loads += 1
        return (polygon,), {(18, 36): (0,)}

    monkeypatch.setattr(m49, "_load_boundaries", load_boundaries)

    first = m49_assignment_for_coordinate(1.0, 1.0)
    second = m49_assignment_for_coordinate(1.0, 1.0)

    assert first == second
    assert loads == 1


def test_missing_m49_crosswalk_remains_explicitly_unassigned() -> None:
    assignment = m49_assignment_for_coordinate(42.65, 21.16)

    assert assignment.stratum == M49_UNASSIGNED
    assert assignment.status == "missing_m49_crosswalk"


def test_row_assignment_uses_the_bbox_midpoint() -> None:
    row = {"bbox_min_x": 2.0, "bbox_max_x": 2.7, "bbox_min_y": 48.5, "bbox_max_y": 49.2}

    assert m49_assignment_for_row(row) == m49_assignment_for_coordinate(48.85, 2.35)


@pytest.mark.parametrize(
    ("field", "value"),
    [("bbox_min_x", 181.0), ("bbox_max_y", float("nan")), ("bbox_min_y", "48.5")],
)
def test_row_assignment_rejects_invalid_coordinates(field: str, value: object) -> None:
    row: dict[str, object] = {
        "bbox_min_x": 2.0,
        "bbox_max_x": 2.7,
        "bbox_min_y": 48.5,
        "bbox_max_y": 49.2,
    }
    row[field] = value

    with pytest.raises(DataValidationError, match=field):
        m49_assignment_for_row(row)


def test_boundary_provenance_pins_dataset_and_stratification_version() -> None:
    provenance = m49_boundary_provenance()

    assert provenance["protocol_version"] == M49_STRATIFICATION_VERSION
    assert provenance["geometry_asset"]["sha256"] == M49_BOUNDARIES_SHA256
    assert provenance["geometry_asset"]["geometry_source_commit"] == (
        "9380cca83db5f9aef52d5e762765100745f84b27"
    )
    assert provenance["geometry_asset"]["geometry_source_sha256"] == (
        "239eec57ac17f100a11e2536cffc56752c318b50ae765b0918ff7aab4ce8f255"
    )
    assert provenance["geometry_asset"]["unmatched_feature_count"] == 16
    assert provenance["geometry_asset"]["m49_source_url"] == (
        "https://unstats.un.org/unsd/methodology/m49/overview/"
    )
    assert (
        "Natural Earth ISO_A3_EH alpha-3 to official UN M49 ISO-alpha3"
        in provenance["geometry_asset"]["geometry_mapping"]
    )


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (True, 0.0),
        (0.0, True),
        ("bad", 0.0),
        (0.0, "bad"),
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (90.1, 0.0),
        (0.0, -180.1),
    ],
)
def test_coordinate_assignment_rejects_invalid_latitude_or_longitude(
    latitude: object, longitude: object
) -> None:
    with pytest.raises(ValueError, match="M49 coordinates must be finite"):
        m49_assignment_for_coordinate(cast(float, latitude), cast(float, longitude))


def test_pinned_asset_checksum_is_checked_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    m49._boundary_bytes.cache_clear()
    m49._load_boundaries.cache_clear()
    monkeypatch.setattr(m49, "M49_BOUNDARIES_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="boundary asset checksum mismatch"):
        m49._boundary_bytes()

    m49._boundary_bytes.cache_clear()
    m49._load_boundaries.cache_clear()


def test_unsupported_geometry_in_the_pinned_asset_fails_closed() -> None:
    feature = {
        "properties": {"m49_subregion_code": "001", "m49_subregion_name": "World"},
        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
    }

    with pytest.raises(RuntimeError, match="unsupported geometry"):
        m49._append_feature_polygons(feature, [])


def test_ring_contains_handles_empty_rings_and_excludes_hole_boundaries() -> None:
    square = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0))

    assert m49._unwrap_ring(()) == ()
    assert m49._ring_contains(square, 1.0, 1.0)
    assert not m49._ring_contains(square, 0.0, 1.0, boundary_is_inside=False)
    assert not m49._ring_contains(square, 3.0, 1.0)


def test_polygon_contains_prepares_boundary_rings_once(monkeypatch: pytest.MonkeyPatch) -> None:
    outer = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0))
    hole = ((4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0))
    polygon = m49._Polygon("015", "Northern Africa", (outer, hole), (0.0, 0.0, 10.0, 10.0))
    unwrap = m49._unwrap_ring
    calls: list[tuple[tuple[float, float], ...]] = []

    def record_unwrap(ring: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
        calls.append(ring)
        return unwrap(ring)

    monkeypatch.setattr(m49, "_unwrap_ring", record_unwrap)

    assert m49._polygon_contains(polygon, 2.0, 2.0)
    assert not m49._polygon_contains(polygon, 5.0, 5.0)
    assert m49._polygon_contains(polygon, 2.0, 3.0)
    assert calls == [outer, hole]
