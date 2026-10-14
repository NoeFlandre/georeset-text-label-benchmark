"""Coordinate assignment to the pinned UN M49 subregion geography."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, cached_property, lru_cache
from importlib.resources import files
from itertools import pairwise
from typing import Any

from georeset_text_label_benchmark.pilot.geo_sampling import bbox_centre

M49_BOUNDARIES_SHA256 = "7be15dfae4eb8741e2518e1bb8bd0e6f2334428ee2fcc9afc659c6377e0907c1"
M49_BOUNDARIES_RESOURCE = "data/un_m49_boundaries.geojson"
M49_STRATIFICATION_VERSION = "un-m49-10m-common-quota-cell-flow-maximin-v2"
M49_UNASSIGNED = "UNASSIGNED"
_GRID_DEGREES = 5
_BOUNDARY_EPSILON = 1e-10
_Ring = tuple[tuple[float, float], ...]
_PreparedRing = tuple[_Ring, float]


@dataclass(frozen=True)
class M49Assignment:
    """The M49 subregion assignment for one coordinate."""

    stratum: str
    subregion_code: str | None
    subregion_name: str | None
    status: str


@dataclass(frozen=True)
class _Polygon:
    subregion_code: str | None
    subregion_name: str | None
    rings: tuple[_Ring, ...]
    bounds: tuple[float, float, float, float]

    @cached_property
    def prepared_rings(self) -> tuple[_PreparedRing, ...]:
        return tuple(_prepare_ring(ring) for ring in self.rings)


def m49_assignment_for_row(row: Mapping[str, Any]) -> M49Assignment:
    """Assign the midpoint of a row's polygon bbox to one UN M49 subregion."""
    return m49_assignment_for_coordinate(*bbox_centre(row))


def m49_assignment_for_coordinate(latitude: float, longitude: float) -> M49Assignment:
    """Assign one unambiguous Natural Earth country hit to its official UN M49 region."""
    _validate_coordinate(latitude, longitude)
    return _cached_assignment_for_coordinate(latitude, longitude, _load_boundaries)


@lru_cache(maxsize=4096)
def _cached_assignment_for_coordinate(
    latitude: float,
    longitude: float,
    load_boundaries: Callable[
        [], tuple[tuple[_Polygon, ...], dict[tuple[int, int], tuple[int, ...]]]
    ],
) -> M49Assignment:
    polygons, index = load_boundaries()
    hits = _coordinate_polygon_hits(latitude, longitude, polygons, index)
    known_hits = {(code, name) for code, name in hits if code is not None}
    has_unmapped_hit = any(code is None for code, _ in hits)
    return _assignment_from_hits(known_hits, has_unmapped_hit)


def _coordinate_polygon_hits(
    latitude: float,
    longitude: float,
    polygons: tuple[_Polygon, ...],
    index: dict[tuple[int, int], tuple[int, ...]],
) -> set[tuple[str | None, str | None]]:
    return {
        (polygon.subregion_code, polygon.subregion_name)
        for polygon in _candidate_polygons(latitude, longitude, polygons, index)
        if _polygon_contains(polygon, latitude, longitude)
    }


def _assignment_from_hits(
    known_hits: set[tuple[str, str | None]], has_unmapped_hit: bool
) -> M49Assignment:
    if _is_unambiguous_known_hit(known_hits, has_unmapped_hit):
        code, name = next(iter(known_hits))
        return _assigned_m49_hit(code, name)
    if _has_ambiguous_m49_hits(known_hits, has_unmapped_hit):
        return M49Assignment(M49_UNASSIGNED, None, None, "ambiguous_boundary")
    if has_unmapped_hit:
        return M49Assignment(M49_UNASSIGNED, None, None, "missing_m49_crosswalk")
    return M49Assignment(M49_UNASSIGNED, None, None, "outside_country_geometry")


def _is_unambiguous_known_hit(
    known_hits: set[tuple[str, str | None]], has_unmapped_hit: bool
) -> bool:
    return len(known_hits) == 1 and not has_unmapped_hit


def _assigned_m49_hit(code: str | None, name: str | None) -> M49Assignment:
    if code is None or name is None:
        raise RuntimeError("pinned M49 crosswalk has incomplete subregion metadata")
    return M49Assignment(code, code, name, "assigned")


def _has_ambiguous_m49_hits(
    known_hits: set[tuple[str, str | None]], has_unmapped_hit: bool
) -> bool:
    return len(known_hits) > 1 or (bool(known_hits) and has_unmapped_hit)


def m49_boundary_provenance() -> dict[str, Any]:
    """Return the pinned geometry and UN crosswalk provenance for run manifests."""
    payload = _boundary_bytes()
    document = json.loads(payload)
    return {
        "protocol_version": M49_STRATIFICATION_VERSION,
        "coordinate": "midpoint of polygon bbox (bbox_min_*, bbox_max_*)",
        "assignment_rule": (
            "Point-in-polygon over all Natural Earth 1:10m Admin-0 features. Assign only "
            "when every containing feature maps to the same UN M49 subregion."
        ),
        "unassigned_policy": (
            "Keep coordinates outside country polygons, on cross-subregion boundaries, or "
            "inside a feature without an M49 crosswalk in the explicit UNASSIGNED stratum; "
            "do not assign to nearest land. Preserve the reason in assignment status."
        ),
        "geometry_asset": {
            "resource": M49_BOUNDARIES_RESOURCE,
            "sha256": M49_BOUNDARIES_SHA256,
            **document["provenance"],
        },
    }


def _validate_coordinate(latitude: float, longitude: float) -> None:
    if not _is_coordinate(latitude, 90) or not _is_coordinate(longitude, 180):
        raise ValueError("M49 coordinates must be finite latitude and longitude values")


def _is_coordinate(value: Any, limit: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and abs(value) <= limit
    )


@cache
def _boundary_bytes() -> bytes:
    payload = files(__package__).joinpath(*M49_BOUNDARIES_RESOURCE.split("/")).read_bytes()
    if hashlib.sha256(payload).hexdigest() != M49_BOUNDARIES_SHA256:
        raise RuntimeError("pinned UN M49 boundary asset checksum mismatch")
    return payload


@cache
def _load_boundaries() -> tuple[tuple[_Polygon, ...], dict[tuple[int, int], tuple[int, ...]]]:
    document = json.loads(_boundary_bytes())
    polygons: list[_Polygon] = []
    for feature in document["features"]:
        _append_feature_polygons(feature, polygons)
    return tuple(polygons), _build_spatial_index(polygons)


def _append_feature_polygons(feature: dict[str, Any], output: list[_Polygon]) -> None:
    properties = feature["properties"]
    geometry = feature["geometry"]
    if geometry["type"] == "Polygon":
        coordinates = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        coordinates = geometry["coordinates"]
    else:
        raise RuntimeError("pinned M49 boundary asset has an unsupported geometry")
    for polygon_rings in coordinates:
        _append_polygon(
            polygon_rings,
            properties["m49_subregion_code"],
            properties["m49_subregion_name"],
            output,
        )


def _append_polygon(
    raw_rings: list[list[list[float]]],
    code: str | None,
    name: str | None,
    output: list[_Polygon],
) -> None:
    rings = tuple(tuple((float(point[0]), float(point[1])) for point in ring) for ring in raw_rings)
    longitudes = [point[0] for point in rings[0]]
    latitudes = [point[1] for point in rings[0]]
    output.append(
        _Polygon(
            code, name, rings, (min(longitudes), min(latitudes), max(longitudes), max(latitudes))
        )
    )


def _build_spatial_index(polygons: list[_Polygon]) -> dict[tuple[int, int], tuple[int, ...]]:
    mutable: dict[tuple[int, int], list[int]] = {}
    for index, polygon in enumerate(polygons):
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        for lat_cell in _cell_range(min_lat, max_lat, -90.0, 90.0):
            for lon_cell in _cell_range(min_lon, max_lon, -180.0, 180.0):
                mutable.setdefault((lat_cell, lon_cell), []).append(index)
    return {key: tuple(indices) for key, indices in mutable.items()}


def _cell_range(minimum: float, maximum: float, origin: float, limit: float) -> range:
    low = max(0, math.floor((minimum - origin) / _GRID_DEGREES))
    high = min(
        math.floor((limit - origin) / _GRID_DEGREES) - 1,
        math.floor((maximum - origin) / _GRID_DEGREES),
    )
    return range(low, high + 1)


def _candidate_polygons(
    latitude: float,
    longitude: float,
    polygons: tuple[_Polygon, ...],
    index: dict[tuple[int, int], tuple[int, ...]],
) -> tuple[_Polygon, ...]:
    lat_cell = min(35, max(0, math.floor((latitude + 90.0) / _GRID_DEGREES)))
    lon_cell = min(71, max(0, math.floor((longitude + 180.0) / _GRID_DEGREES)))
    return tuple(polygons[polygon_index] for polygon_index in index.get((lat_cell, lon_cell), ()))


def _polygon_contains(polygon: _Polygon, latitude: float, longitude: float) -> bool:
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    if not (min_lat - _BOUNDARY_EPSILON <= latitude <= max_lat + _BOUNDARY_EPSILON):
        return False
    adjusted = _longitude_near(longitude, (min_lon + max_lon) / 2)
    if not _ring_contains_prepared(polygon.prepared_rings[0], latitude, adjusted):
        return False
    return not any(
        _ring_contains_prepared(ring, latitude, adjusted, boundary_is_inside=False)
        for ring in polygon.prepared_rings[1:]
    )


def _longitude_near(longitude: float, reference: float) -> float:
    return longitude + 360.0 * round((reference - longitude) / 360.0)


def _ring_contains(
    ring: tuple[tuple[float, float], ...],
    latitude: float,
    longitude: float,
    *,
    boundary_is_inside: bool = True,
) -> bool:
    return _ring_contains_prepared(
        _prepare_ring(ring), latitude, longitude, boundary_is_inside=boundary_is_inside
    )


def _prepare_ring(ring: _Ring) -> _PreparedRing:
    points = _unwrap_ring(ring)
    reference = sum(point[0] for point in points) / len(points)
    return points, reference


def _ring_contains_prepared(
    prepared_ring: _PreparedRing,
    latitude: float,
    longitude: float,
    *,
    boundary_is_inside: bool = True,
) -> bool:
    points, reference = prepared_ring
    adjusted_longitude = _longitude_near(longitude, reference)
    inside = False
    for first, second in pairwise(points):
        if _point_on_segment(first, second, adjusted_longitude, latitude):
            return boundary_is_inside
        if _segment_crosses_ray(first, second, adjusted_longitude, latitude):
            inside = not inside
    return inside


def _segment_crosses_ray(
    first: tuple[float, float],
    second: tuple[float, float],
    longitude: float,
    latitude: float,
) -> bool:
    if (first[1] > latitude) == (second[1] > latitude):
        return False
    crossing = (second[0] - first[0]) * (latitude - first[1]) / (second[1] - first[1]) + first[0]
    return longitude < crossing


def _unwrap_ring(ring: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    if not ring:
        return ()
    points = [ring[0]]
    for longitude, latitude in ring[1:]:
        previous = points[-1][0]
        points.append((_longitude_near(longitude, previous), latitude))
    return tuple(points)


def _point_on_segment(
    first: tuple[float, float],
    second: tuple[float, float],
    longitude: float,
    latitude: float,
) -> bool:
    x1, y1 = first
    x2, y2 = second
    cross = (latitude - y1) * (x2 - x1) - (longitude - x1) * (y2 - y1)
    if abs(cross) > _BOUNDARY_EPSILON:
        return False
    return (
        min(x1, x2) - _BOUNDARY_EPSILON <= longitude <= max(x1, x2) + _BOUNDARY_EPSILON
        and min(y1, y2) - _BOUNDARY_EPSILON <= latitude <= max(y1, y2) + _BOUNDARY_EPSILON
    )
