"""Deterministic, geographically spread sample of yes and no EUNIS sentences.

Coordinates are the midpoint of each polygon's bbox. Cells are H3 resolution 3. Each
group (yes, no) takes distinct cells, one sentence per cell. Cells are chosen with
maximin spacing, as in the landuse-sentence-relevance-golden-human-set sampler.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import h3

from georeset_text_label_benchmark.errors import DataValidationError

H3_RESOLUTION = 3
GROUPS = ("yes", "no")
ENGLISH = "eng"
EARTH_RADIUS_KM = 6371.0088
_OCCURRENCE_FIELDS = (
    "source_pbf",
    "osm_type",
    "osm_id",
    "description_identity",
    "tag_key",
    "sentence_index",
    "text_sha256",
)

CellOf = Callable[[float, float], str]
CentreOf = Callable[[str], tuple[float, float]]


def h3_cell_of(lat: float, lon: float) -> str:
    """Return the resolution-3 H3 cell that contains a latitude and longitude."""
    return h3.latlng_to_cell(lat, lon, H3_RESOLUTION)


def h3_centre_of(cell: str) -> tuple[float, float]:
    """Return the latitude and longitude of an H3 cell centre."""
    lat, lon = h3.cell_to_latlng(cell)
    return float(lat), float(lon)


def bbox_centre(row: Mapping[str, Any]) -> tuple[float, float]:
    """Return the (latitude, longitude) midpoint of a polygon bbox."""
    min_x = _coordinate(row, "bbox_min_x", 180.0)
    min_y = _coordinate(row, "bbox_min_y", 90.0)
    max_x = _coordinate(row, "bbox_max_x", 180.0)
    max_y = _coordinate(row, "bbox_max_y", 90.0)
    return (min_y + max_y) / 2.0, (min_x + max_x) / 2.0


def select_geographic_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_of: CellOf,
    centre_of: CentreOf,
    per_group: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Choose ``per_group`` yes and ``per_group`` no sentences on distinct, spread cells."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    unique = _deduplicate(_eligible(rows), seed)
    by_group = _candidates_by_cell(unique, cell_of)
    _require_cells(by_group, per_group)
    centres = {cell: centre_of(cell) for group in GROUPS for cell in by_group[group]}
    chosen = _choose_cells(by_group, centres, per_group, seed)
    return [
        {
            **by_group[group][cell][0],
            "h3_cell": cell,
            "cell_centre_lat": centres[cell][0],
            "cell_centre_lon": centres[cell][1],
        }
        for group, cell in chosen
    ]


def _require_cells(by_group: Mapping[str, Mapping[str, Any]], per_group: int) -> None:
    for group in GROUPS:
        if len(by_group[group]) < per_group:
            raise ValueError(f"need {per_group} {group} cells, found {len(by_group[group])}")


def _coordinate(row: Mapping[str, Any], field: str, limit: float) -> float:
    value = row.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or abs(value) > limit
    ):
        raise DataValidationError(f"polygon {field} must be a finite coordinate within ±{limit:g}")
    return float(value)


def _eligible(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("language_code") != ENGLISH or row.get("eunis_code") is None:
            continue
        if row.get("decision") not in GROUPS:
            raise ValueError(f"unknown decision {row.get('decision')!r}")
        eligible.append(row)
    return eligible


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _occurrence_id(row: Mapping[str, Any]) -> str:
    return "|".join(str(row[field]) for field in _OCCURRENCE_FIELDS)


def _deduplicate(rows: list[Mapping[str, Any]], seed: int) -> list[Mapping[str, Any]]:
    """Keep one occurrence per sentence text and per polygon, in seeded order."""
    ordered = sorted(rows, key=lambda row: _stable_key(seed, _occurrence_id(row)))
    seen_text: set[str] = set()
    seen_polygons: set[tuple[str, str, int]] = set()
    kept: list[Mapping[str, Any]] = []
    for row in ordered:
        polygon = (row["source_pbf"], row["osm_type"], row["osm_id"])
        if row["text_sha256"] in seen_text or polygon in seen_polygons:
            continue
        seen_text.add(row["text_sha256"])
        seen_polygons.add(polygon)
        kept.append(row)
    return kept


def _candidates_by_cell(
    rows: Sequence[Mapping[str, Any]], cell_of: CellOf
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    """Group rows by group then cell. Each cell list keeps the seeded order."""
    by_group: dict[str, dict[str, list[Mapping[str, Any]]]] = {group: {} for group in GROUPS}
    for row in rows:
        lat, lon = bbox_centre(row)
        by_group[row["decision"]].setdefault(cell_of(lat, lon), []).append(row)
    return by_group


def _choose_cells(
    by_group: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    centres: Mapping[str, tuple[float, float]],
    per_group: int,
    seed: int,
) -> list[tuple[str, str]]:
    available = {group: set(by_group[group]) for group in GROUPS}
    selected: dict[str, list[str]] = {group: [] for group in GROUPS}
    nearest: dict[str, float] = {}
    order: list[tuple[str, str]] = []
    for _ in range(per_group):
        for group in GROUPS:
            cell = _next_cell(group, available, selected, per_group, nearest, seed)
            _take(cell, group, available, selected, nearest, centres)
            order.append((group, cell))
    return order


def _next_cell(
    group: str,
    available: Mapping[str, set[str]],
    selected: Mapping[str, list[str]],
    per_group: int,
    nearest: Mapping[str, float],
    seed: int,
) -> str:
    candidates = [
        cell
        for cell in sorted(available[group])
        if _leaves_groups_feasible(cell, group, available, selected, per_group)
    ]
    if not candidates:
        raise ValueError(f"cannot choose {per_group} disjoint cells for {group}")
    if not nearest:
        return min(candidates, key=lambda cell: _stable_key(seed, cell))
    return max(candidates, key=lambda cell: (nearest[cell], _stable_key(seed, cell)))


def _leaves_groups_feasible(
    cell: str,
    group: str,
    available: Mapping[str, set[str]],
    selected: Mapping[str, list[str]],
    per_group: int,
) -> bool:
    for other in GROUPS:
        remaining = len(available[other]) - (1 if cell in available[other] else 0)
        required = per_group - len(selected[other]) - (1 if other == group else 0)
        if remaining < required:
            return False
    return True


def _take(
    cell: str,
    group: str,
    available: dict[str, set[str]],
    selected: dict[str, list[str]],
    nearest: dict[str, float],
    centres: Mapping[str, tuple[float, float]],
) -> None:
    for other in GROUPS:
        available[other].discard(cell)
    selected[group].append(cell)
    anchor = centres[cell]
    for other_cell in {cell for group_cells in available.values() for cell in group_cells}:
        distance = _haversine_km(centres[other_cell], anchor)
        nearest[other_cell] = min(nearest.get(other_cell, math.inf), distance)


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat_1, lon_1 = map(math.radians, first)
    lat_2, lon_2 = map(math.radians, second)
    haversine = (
        math.sin((lat_2 - lat_1) / 2) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin((lon_2 - lon_1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(haversine))
