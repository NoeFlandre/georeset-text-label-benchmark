"""Deterministic, geographically spread sample of yes and no EUNIS sentences.

Coordinates are the midpoint of each polygon's bbox. Cells are H3 resolution 3. Each
group (yes, no) takes distinct cells, one sentence per cell. Cells are chosen with
maximin spacing, as in the landuse-sentence-relevance-golden-human-set sampler.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
StratumOf = Callable[[Mapping[str, Any]], str]


def h3_cell_of(lat: float, lon: float) -> str:
    """Return the resolution-3 H3 cell that contains a latitude and longitude."""
    return h3.latlng_to_cell(lat, lon, H3_RESOLUTION)


def h3_centre_of(cell: str) -> tuple[float, float]:
    """Return the latitude and longitude of an H3 cell centre."""
    lat, lon = h3.cell_to_latlng(cell)
    return float(lat), float(lon)


def sample_id_of(row: Mapping[str, Any]) -> str:
    """Return a stable identifier for one sentence occurrence."""
    identity = [row[field] for field in _OCCURRENCE_FIELDS]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        polygon = _polygon_key(row)
        if row["text_sha256"] in seen_text or polygon in seen_polygons:
            continue
        seen_text.add(row["text_sha256"])
        seen_polygons.add(polygon)
        kept.append(row)
    return kept


def _polygon_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return row["source_pbf"], row["osm_type"], row["osm_id"]


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


def extend_geographic_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_of: CellOf,
    centre_of: CentreOf,
    existing: Sequence[Mapping[str, Any]],
    per_group: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Add rows by group maximin spacing while keeping the existing sample unchanged."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    _validate_existing_sample(existing, per_group)
    eligible = _deduplicate(_eligible(rows), seed)
    candidates = _remove_existing_identities(eligible, existing)
    by_group = _candidates_by_cell(candidates, cell_of)
    added: list[dict[str, Any]] = []
    for group in GROUPS:
        added.extend(_extend_group(group, by_group[group], existing, per_group, centre_of, seed))
    return added


def extend_region_balanced_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_of: CellOf,
    centre_of: CentreOf,
    stratum_of: StratumOf,
    existing: Sequence[Mapping[str, Any]],
    per_group: int,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extend a frozen sample with common quotas constrained by cell-to-region flow."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    _validate_existing_sample(existing, per_group)
    eligible = _deduplicate(_eligible(rows), seed)
    candidates = _remove_existing_identities(eligible, existing)
    by_group_region = _candidate_region_rows_by_group(candidates, cell_of, stratum_of, existing)
    prefix_counts = _existing_stratum_counts(existing, stratum_of)
    row_capacities, candidate_capacities, unique_cells, multi_region_cells = (
        _candidate_region_capacities(by_group_region)
    )
    quotas = _common_stratum_quotas(
        prefix_counts,
        by_group_region,
        per_group,
        seed,
        row_capacities,
        candidate_capacities,
    )
    added = _extend_flow_by_region(by_group_region, existing, quotas, stratum_of, centre_of, seed)
    _require_extension_fill(added, existing, per_group)
    return added, {
        "target_counts_per_group": dict(sorted(quotas.items())),
        "prefix_counts_by_decision": _format_group_stratum_counts(prefix_counts),
        "candidate_row_capacity_by_decision": _format_group_stratum_counts(row_capacities),
        "candidate_cell_capacity_by_decision": _format_group_stratum_counts(candidate_capacities),
        "candidate_unique_cell_count_by_decision": unique_cells,
        "candidate_multi_region_cell_count_by_decision": multi_region_cells,
        "cell_capacities_overlap_across_regions": True,
        "selected_counts_by_decision": _format_group_stratum_counts(
            _selected_stratum_counts(existing, added, stratum_of)
        ),
    }


def _candidate_region_rows_by_group(
    rows: Sequence[Mapping[str, Any]],
    cell_of: CellOf,
    stratum_of: StratumOf,
    existing: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, list[Mapping[str, Any]]]]]:
    """Keep every cell-to-region edge represented by a candidate row."""
    output: dict[str, dict[str, dict[str, list[Mapping[str, Any]]]]] = {
        group: {} for group in GROUPS
    }
    used_cells = _existing_cells_by_group(existing)
    for row in rows:
        _add_region_candidate(row, cell_of, stratum_of, used_cells, output)
    return output


def _existing_cells_by_group(existing: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    return {
        group: {row["h3_cell"] for row in existing if row["decision"] == group} for group in GROUPS
    }


def _add_region_candidate(
    row: Mapping[str, Any],
    cell_of: CellOf,
    stratum_of: StratumOf,
    used_cells: dict[str, set[str]],
    output: dict[str, dict[str, dict[str, list[Mapping[str, Any]]]]],
) -> None:
    lat, lon = bbox_centre(row)
    cell = cell_of(lat, lon)
    group = row["decision"]
    if cell in used_cells[group]:
        return
    stratum = _validated_stratum(stratum_of(row))
    output[group].setdefault(cell, {}).setdefault(stratum, []).append(row)


def _validated_stratum(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("geographic stratum must be a non-empty string")
    return value


def _candidate_region_capacities(
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    row_counts = _empty_group_stratum_counts()
    cell_counts = _empty_group_stratum_counts()
    unique_cells = _unique_candidate_cell_counts(candidates)
    multi_region_cells = _multi_region_candidate_cell_counts(candidates)
    _count_all_candidate_regions(candidates, row_counts, cell_counts)
    return row_counts, cell_counts, unique_cells, multi_region_cells


def _empty_group_stratum_counts() -> dict[str, dict[str, int]]:
    return {group: {} for group in GROUPS}


def _unique_candidate_cell_counts(
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    return {group: len(candidates[group]) for group in GROUPS}


def _multi_region_candidate_cell_counts(
    candidates: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, int]:
    return {
        group: sum(len(regions) > 1 for regions in candidates[group].values()) for group in GROUPS
    }


def _count_all_candidate_regions(
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    row_counts: dict[str, dict[str, int]],
    cell_counts: dict[str, dict[str, int]],
) -> None:
    for group in GROUPS:
        _count_group_candidate_regions(group, candidates[group], row_counts, cell_counts)


def _count_group_candidate_regions(
    group: str,
    candidates: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    row_counts: dict[str, dict[str, int]],
    cell_counts: dict[str, dict[str, int]],
) -> None:
    for regions in candidates.values():
        _count_candidate_region_rows(group, regions, row_counts, cell_counts)


def _count_candidate_region_rows(
    group: str,
    regions: Mapping[str, Sequence[Mapping[str, Any]]],
    row_counts: dict[str, dict[str, int]],
    cell_counts: dict[str, dict[str, int]],
) -> None:
    for stratum, rows in regions.items():
        row_counts[group][stratum] = row_counts[group].get(stratum, 0) + len(rows)
        cell_counts[group][stratum] = cell_counts[group].get(stratum, 0) + 1


def _existing_stratum_counts(
    existing: Sequence[Mapping[str, Any]], stratum_of: StratumOf
) -> dict[str, dict[str, int]]:
    counts = {group: {} for group in GROUPS}
    for row in existing:
        group = row["decision"]
        stratum = _validated_stratum(stratum_of(row))
        counts[group][stratum] = counts[group].get(stratum, 0) + 1
    return counts


def _common_stratum_quotas(
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    per_group: int,
    seed: int,
    row_capacities: Mapping[str, Mapping[str, int]],
    cell_capacities: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    strata = _observed_strata(prefix, candidates)
    quotas = _minimum_common_quotas(strata, prefix)
    if (
        sum(quotas.values()) > per_group
        or _global_candidate_capacity_is_insufficient(per_group, prefix, candidates)
        or not _common_quota_is_feasible(quotas, prefix, candidates)
    ):
        raise _region_capacity_error(
            per_group, quotas, prefix, candidates, row_capacities, cell_capacities
        )
    maximums = _maximum_common_quota_bounds(strata, prefix, cell_capacities)
    completed = _search_common_quota_completion(
        quotas,
        strata,
        maximums,
        prefix,
        candidates,
        per_group,
        seed,
        set(),
    )
    if completed is None:
        raise _region_capacity_error(
            per_group, quotas, prefix, candidates, row_capacities, cell_capacities
        )
    return dict(sorted(completed.items()))


def _observed_strata(
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> list[str]:
    observed = set(prefix["yes"]) | set(prefix["no"])
    for group in GROUPS:
        observed.update(_candidate_strata(candidates[group]))
    return sorted(observed)


def _candidate_strata(
    candidates: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> set[str]:
    return {stratum for regions in candidates.values() for stratum in regions}


def _minimum_common_quotas(
    strata: Sequence[str], prefix: Mapping[str, Mapping[str, int]]
) -> dict[str, int]:
    return {stratum: max(prefix[group].get(stratum, 0) for group in GROUPS) for stratum in strata}


def _global_candidate_capacity_is_insufficient(
    per_group: int,
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> bool:
    return any(len(candidates[group]) < per_group - sum(prefix[group].values()) for group in GROUPS)


def _common_quota_is_feasible(
    quotas: Mapping[str, int],
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> bool:
    return all(_group_quota_is_feasible(group, quotas, prefix, candidates) for group in GROUPS)


def _group_quota_is_feasible(
    group: str,
    quotas: Mapping[str, int],
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> bool:
    demands = _regional_demands(quotas, prefix[group])
    return _region_flow_value(candidates[group], demands) == sum(demands.values())


def _regional_demands(quotas: Mapping[str, int], prefix: Mapping[str, int]) -> dict[str, int]:
    return {stratum: quota - prefix.get(stratum, 0) for stratum, quota in quotas.items()}


def _maximum_common_quota_bounds(
    strata: Sequence[str],
    prefix: Mapping[str, Mapping[str, int]],
    cell_capacities: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    return {
        stratum: min(
            prefix[group].get(stratum, 0) + cell_capacities[group].get(stratum, 0)
            for group in GROUPS
        )
        for stratum in strata
    }


def _search_common_quota_completion(
    quotas: dict[str, int],
    strata: Sequence[str],
    maximums: Mapping[str, int],
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    per_group: int,
    seed: int,
    failed_states: set[tuple[int, ...]],
) -> dict[str, int] | None:
    initial_state = _quota_state(quotas, strata)
    if sum(initial_state) == per_group:
        return quotas
    if _quota_search_state_is_exhausted(
        initial_state,
        failed_states,
        quotas,
        strata,
        maximums,
        per_group - sum(initial_state),
    ):
        failed_states.add(initial_state)
        return None
    return _search_common_quota_stack(
        [_quota_search_frame(initial_state, strata, seed)],
        strata,
        maximums,
        prefix,
        candidates,
        per_group,
        seed,
        failed_states,
    )


def _quota_state(quotas: Mapping[str, int], strata: Sequence[str]) -> tuple[int, ...]:
    return tuple(quotas[stratum] for stratum in strata)


def _quota_counts(state: tuple[int, ...], strata: Sequence[str]) -> dict[str, int]:
    return dict(zip(strata, state, strict=True))


def _quota_search_frame(
    state: tuple[int, ...], strata: Sequence[str], seed: int
) -> _QuotaSearchFrame:
    quotas = _quota_counts(state, strata)
    ordered = sorted(strata, key=lambda item: (quotas[item], _stable_key(seed, item)))
    return _QuotaSearchFrame(state, ordered)


def _quota_search_state_is_exhausted(
    state: tuple[int, ...],
    failed_states: set[tuple[int, ...]],
    quotas: Mapping[str, int],
    strata: Sequence[str],
    maximums: Mapping[str, int],
    remaining: int,
) -> bool:
    return state in failed_states or _quota_upper_bounds_cannot_fill(
        quotas, strata, maximums, remaining
    )


def _search_common_quota_stack(
    stack: list[_QuotaSearchFrame],
    strata: Sequence[str],
    maximums: Mapping[str, int],
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    per_group: int,
    seed: int,
    failed_states: set[tuple[int, ...]],
) -> dict[str, int] | None:
    while stack:
        frame = stack[-1]
        trial = _next_quota_trial(frame, strata, maximums)
        if trial is None:
            if frame.next_index == len(frame.ordered_strata):
                failed_states.add(frame.state)
                stack.pop()
            continue
        completed = _accept_quota_trial(
            trial,
            strata,
            maximums,
            prefix,
            candidates,
            per_group,
            seed,
            failed_states,
            stack,
        )
        if completed is not None:
            return completed
    return None


def _next_quota_trial(
    frame: _QuotaSearchFrame, strata: Sequence[str], maximums: Mapping[str, int]
) -> tuple[int, ...] | None:
    if frame.next_index >= len(frame.ordered_strata):
        return None
    stratum = frame.ordered_strata[frame.next_index]
    frame.next_index += 1
    index = strata.index(stratum)
    if frame.state[index] >= maximums[stratum]:
        return None
    trial = list(frame.state)
    trial[index] += 1
    return tuple(trial)


def _accept_quota_trial(
    trial: tuple[int, ...],
    strata: Sequence[str],
    maximums: Mapping[str, int],
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    per_group: int,
    seed: int,
    failed_states: set[tuple[int, ...]],
    stack: list[_QuotaSearchFrame],
) -> dict[str, int] | None:
    if trial in failed_states:
        return None
    quotas = _quota_counts(trial, strata)
    if not _common_quota_is_feasible(quotas, prefix, candidates):
        failed_states.add(trial)
        return None
    total = sum(trial)
    if total == per_group:
        return quotas
    if _quota_upper_bounds_cannot_fill(quotas, strata, maximums, per_group - total):
        failed_states.add(trial)
        return None
    stack.append(_quota_search_frame(trial, strata, seed))
    return None


def _quota_upper_bounds_cannot_fill(
    quotas: Mapping[str, int],
    strata: Sequence[str],
    maximums: Mapping[str, int],
    remaining: int,
) -> bool:
    return sum(maximums[stratum] - quotas[stratum] for stratum in strata) < remaining


def _region_capacity_error(
    per_group: int,
    quotas: Mapping[str, int],
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    row_capacities: Mapping[str, Mapping[str, int]],
    cell_capacities: Mapping[str, Mapping[str, int]],
) -> ValueError:
    detail = _region_capacity_error_details(
        per_group, quotas, prefix, candidates, row_capacities, cell_capacities
    )
    return ValueError(f"cannot meet common regional quotas: {json.dumps(detail, sort_keys=True)}")


def _region_capacity_error_details(
    per_group: int,
    quotas: Mapping[str, int],
    prefix: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    row_capacities: Mapping[str, Mapping[str, int]],
    cell_capacities: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    unique_cells, multi_region_cells = _candidate_cell_summary(candidates)
    lower, upper = _common_region_prefix_bounds(prefix, cell_capacities, quotas)
    strata = sorted(lower)
    return {
        "per_group": per_group,
        "prefix_counts_by_decision": _format_group_stratum_counts(prefix),
        "candidate_row_capacity_by_decision": _format_group_stratum_counts(row_capacities),
        "candidate_cell_capacity_by_decision": _format_group_stratum_counts(cell_capacities),
        "candidate_unique_cell_count_by_decision": unique_cells,
        "candidate_multi_region_cell_count_by_decision": multi_region_cells,
        "additional_flow_capacity_by_decision": _candidate_flow_capacities(candidates, per_group),
        "minimum_common_quota_by_stratum": dict(sorted(quotas.items())),
        "unmatchable_prefix_strata": _unmatchable_prefix_strata(strata, lower, upper),
        "allocated_common_total": sum(quotas.values()),
        "shortfall_per_group": max(0, per_group - sum(quotas.values())),
        "cell_capacities_overlap_across_regions": True,
    }


def _candidate_cell_summary(
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> tuple[dict[str, int], dict[str, int]]:
    return (
        _unique_candidate_cell_counts(candidates),
        _multi_region_candidate_cell_counts(candidates),
    )


def _common_region_prefix_bounds(
    prefix: Mapping[str, Mapping[str, int]],
    cell_capacities: Mapping[str, Mapping[str, int]],
    quotas: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    strata = sorted(set(prefix["yes"]) | set(prefix["no"]) | set(quotas))
    lower: dict[str, int] = {}
    upper: dict[str, int] = {}
    for stratum in strata:
        lower[stratum] = max(prefix[group].get(stratum, 0) for group in GROUPS)
        upper[stratum] = min(
            prefix[group].get(stratum, 0) + cell_capacities[group].get(stratum, 0)
            for group in GROUPS
        )
    return lower, upper


def _unmatchable_prefix_strata(
    strata: Sequence[str], lower: Mapping[str, int], upper: Mapping[str, int]
) -> dict[str, list[int]]:
    return {
        stratum: [lower[stratum], upper[stratum]]
        for stratum in strata
        if lower[stratum] > upper[stratum]
    }


def _candidate_flow_capacities(
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    per_group: int,
) -> dict[str, int]:
    return {
        group: _region_flow_value(
            candidates[group],
            dict.fromkeys(_candidate_strata(candidates[group]), per_group),
        )
        for group in GROUPS
    }


def _region_flow_value(
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    demands: Mapping[str, int],
    excluded_cells: set[str] | None = None,
) -> int:
    required = sum(demands.values())
    if required <= 0:
        return 0
    graph, source, sink, _available_cells = _region_flow_graph(cells, demands, excluded_cells)
    return _maximum_flow(graph, source, sink, required)


def _region_flow_graph(
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    demands: Mapping[str, int],
    excluded_cells: set[str] | None,
) -> tuple[list[list[_FlowEdge]], int, int, list[str]]:
    regions = _required_flow_regions(demands)
    available_cells = _available_flow_cells(cells, regions, excluded_cells)
    graph, source, sink, region_nodes = _flow_graph_nodes(len(available_cells), regions)
    _add_region_flow_edges(graph, cells, demands, available_cells, region_nodes, source, sink)
    return graph, source, sink, available_cells


def _required_flow_regions(demands: Mapping[str, int]) -> list[str]:
    return [stratum for stratum, demand in sorted(demands.items()) if demand > 0]


def _available_flow_cells(
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    regions: Sequence[str],
    excluded_cells: set[str] | None,
) -> list[str]:
    return sorted(
        cell
        for cell, cell_regions in cells.items()
        if _is_available_flow_cell(cell, cell_regions, regions, excluded_cells)
    )


def _is_available_flow_cell(
    cell: str,
    cell_regions: Mapping[str, Any],
    regions: Sequence[str],
    excluded_cells: set[str] | None,
) -> bool:
    return (excluded_cells is None or cell not in excluded_cells) and any(
        stratum in regions for stratum in cell_regions
    )


def _flow_graph_nodes(
    cell_count: int, regions: Sequence[str]
) -> tuple[list[list[_FlowEdge]], int, int, dict[str, int]]:
    source = 0
    cell_start = 1
    region_start = cell_start + cell_count
    sink = region_start + len(regions)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    region_nodes = {stratum: region_start + index for index, stratum in enumerate(regions)}
    return graph, source, sink, region_nodes


def _add_region_flow_edges(
    graph: list[list[_FlowEdge]],
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    demands: Mapping[str, int],
    available_cells: Sequence[str],
    region_nodes: Mapping[str, int],
    source: int,
    sink: int,
) -> None:
    for index, cell in enumerate(available_cells):
        cell_node = index + 1
        _add_flow_edge(graph, source, cell_node, 1)
        for stratum in sorted(cells[cell]):
            if stratum in region_nodes:
                _add_flow_edge(graph, cell_node, region_nodes[stratum], 1)
    for stratum, node in region_nodes.items():
        _add_flow_edge(graph, node, sink, demands[stratum])


@dataclass
class _FlowEdge:
    target: int
    residual: int
    reverse_index: int


@dataclass
class _QuotaSearchFrame:
    state: tuple[int, ...]
    ordered_strata: list[str]
    next_index: int = 0


def _add_flow_edge(graph: list[list[_FlowEdge]], source: int, target: int, capacity: int) -> None:
    forward = _FlowEdge(target, capacity, len(graph[target]))
    reverse = _FlowEdge(source, 0, len(graph[source]))
    graph[source].append(forward)
    graph[target].append(reverse)


def _maximum_flow(graph: list[list[_FlowEdge]], source: int, sink: int, limit: int) -> int:
    flow = 0
    while flow < limit:
        levels = _flow_levels(graph, source)
        if levels[sink] < 0:
            return flow
        cursors = [0] * len(graph)
        flow = _send_available_flow(graph, levels, cursors, source, sink, flow, limit)
    return flow


def _flow_levels(graph: list[list[_FlowEdge]], source: int) -> list[int]:
    levels = [-1] * len(graph)
    levels[source] = 0
    pending = deque([source])
    while pending:
        node = pending.popleft()
        for edge in graph[node]:
            if edge.residual and levels[edge.target] < 0:
                levels[edge.target] = levels[node] + 1
                pending.append(edge.target)
    return levels


def _send_available_flow(
    graph: list[list[_FlowEdge]],
    levels: Sequence[int],
    cursors: list[int],
    node: int,
    sink: int,
    flow: int,
    limit: int,
) -> int:
    while flow < limit:
        sent = _send_one_flow_path(graph, levels, cursors, node, sink, limit - flow)
        if sent == 0:
            return flow
        flow += sent
    return flow


def _send_one_flow_path(
    graph: list[list[_FlowEdge]],
    levels: Sequence[int],
    cursors: list[int],
    node: int,
    sink: int,
    amount: int,
) -> int:
    if node == sink:
        return amount
    while cursors[node] < len(graph[node]):
        sent = _send_flow_edge_at_cursor(graph, levels, cursors, node, sink, amount)
        if sent > 0:
            return sent
        cursors[node] += 1
    return 0


def _send_flow_edge_at_cursor(
    graph: list[list[_FlowEdge]],
    levels: Sequence[int],
    cursors: list[int],
    node: int,
    sink: int,
    amount: int,
) -> int:
    edge = graph[node][cursors[node]]
    if edge.residual <= 0 or levels[edge.target] != levels[node] + 1:
        return 0
    sent = _send_one_flow_path(
        graph, levels, cursors, edge.target, sink, min(amount, edge.residual)
    )
    if sent == 0:
        return 0
    edge.residual -= sent
    graph[edge.target][edge.reverse_index].residual += sent
    return sent


def _extend_flow_by_region(
    candidates: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    existing: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
    stratum_of: StratumOf,
    centre_of: CentreOf,
    seed: int,
) -> list[dict[str, Any]]:
    added: list[dict[str, Any]] = []
    for group in GROUPS:
        added.extend(
            _extend_flow_group(
                group, candidates[group], existing, quotas, stratum_of, centre_of, seed
            )
        )
    return added


def _extend_flow_group(
    group: str,
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    existing: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
    stratum_of: StratumOf,
    centre_of: CentreOf,
    seed: int,
) -> list[dict[str, Any]]:
    prefix = _flow_group_prefix(existing, group)
    prefix_counts = _existing_stratum_counts(prefix, stratum_of)[group]
    demands = _regional_demands(quotas, prefix_counts)
    _require_group_flow_capacity(group, cells, demands)
    centres = _flow_candidate_centres(cells, centre_of)
    anchors = _prefix_centres_by_region(prefix, stratum_of)
    selected_cells: set[str] = set()
    selected_counts: dict[str, int] = dict.fromkeys(demands, 0)
    added: list[dict[str, Any]] = []
    while any(demand > 0 for demand in demands.values()):
        _select_next_flow_cell(
            cells,
            demands,
            prefix_counts,
            selected_counts,
            selected_cells,
            centres,
            anchors,
            added,
            seed,
        )
    return added


def _flow_group_prefix(
    existing: Sequence[Mapping[str, Any]], group: str
) -> list[Mapping[str, Any]]:
    return [row for row in existing if row["decision"] == group]


def _flow_candidate_centres(
    cells: Mapping[str, Any], centre_of: CentreOf
) -> dict[str, tuple[float, float]]:
    return {cell: centre_of(cell) for cell in cells}


def _require_group_flow_capacity(
    group: str,
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    demands: Mapping[str, int],
) -> None:
    if _region_flow_value(cells, demands) != sum(demands.values()):
        raise ValueError(f"M49 flow allocation is infeasible for {group}")


def _select_next_flow_cell(
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    demands: dict[str, int],
    prefix_counts: Mapping[str, int],
    selected_counts: dict[str, int],
    selected_cells: set[str],
    centres: Mapping[str, tuple[float, float]],
    anchors: dict[str, list[tuple[float, float]]],
    added: list[dict[str, Any]],
    seed: int,
) -> None:
    stratum = _least_filled_region(prefix_counts, selected_counts, demands, seed)
    cell = _maximin_feasible_cell(stratum, cells, demands, selected_cells, centres, anchors, seed)
    added.append(_row_for_extension_cell(cells[cell][stratum][0], cell, centres[cell]))
    selected_cells.add(cell)
    demands[stratum] -= 1
    selected_counts[stratum] += 1
    anchors.setdefault(stratum, []).append(centres[cell])


def _prefix_centres_by_region(
    prefix: Sequence[Mapping[str, Any]], stratum_of: StratumOf
) -> dict[str, list[tuple[float, float]]]:
    anchors: dict[str, list[tuple[float, float]]] = {}
    for row in prefix:
        stratum = _validated_stratum(stratum_of(row))
        anchors.setdefault(stratum, []).append((row["cell_centre_lat"], row["cell_centre_lon"]))
    return anchors


def _least_filled_region(
    prefix_counts: Mapping[str, int],
    selected_counts: Mapping[str, int],
    demands: Mapping[str, int],
    seed: int,
) -> str:
    available = [stratum for stratum, demand in demands.items() if demand > 0]
    return min(
        available,
        key=lambda stratum: (
            prefix_counts.get(stratum, 0) + selected_counts[stratum],
            _stable_key(seed, stratum),
        ),
    )


def _maximin_feasible_cell(
    stratum: str,
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    demands: Mapping[str, int],
    selected_cells: set[str],
    centres: Mapping[str, tuple[float, float]],
    anchors: Mapping[str, Sequence[tuple[float, float]]],
    seed: int,
) -> str:
    ranked = _rank_maximin_candidates(stratum, cells, selected_cells, centres, anchors, seed)
    return _first_flow_feasible_candidate(stratum, ranked, cells, demands, selected_cells)


def _rank_maximin_candidates(
    stratum: str,
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    selected_cells: set[str],
    centres: Mapping[str, tuple[float, float]],
    anchors: Mapping[str, Sequence[tuple[float, float]]],
    seed: int,
) -> list[str]:
    candidates = [
        cell for cell, regions in cells.items() if cell not in selected_cells and stratum in regions
    ]
    return sorted(
        candidates,
        key=lambda cell: (
            -_nearest_anchor_distance(centres[cell], anchors.get(stratum, ())),
            _stable_key(seed, cell),
        ),
    )


def _first_flow_feasible_candidate(
    stratum: str,
    ranked: Sequence[str],
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    demands: Mapping[str, int],
    selected_cells: set[str],
) -> str:
    for cell in ranked:
        if _candidate_preserves_flow(stratum, cell, cells, demands, selected_cells):
            return cell
    raise ValueError(f"M49 maximin selection exhausted feasible {stratum} cells")


def _candidate_preserves_flow(
    stratum: str,
    cell: str,
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    demands: Mapping[str, int],
    selected_cells: set[str],
) -> bool:
    trial_demands = dict(demands)
    trial_demands[stratum] -= 1
    excluded = {*selected_cells, cell}
    return _region_flow_value(cells, trial_demands, excluded) == sum(trial_demands.values())


def _nearest_anchor_distance(
    centre: tuple[float, float], anchors: Sequence[tuple[float, float]]
) -> float:
    return min((_haversine_km(centre, anchor) for anchor in anchors), default=math.inf)


def _require_extension_fill(
    added: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
    per_group: int,
) -> None:
    expected = len(GROUPS) * per_group - len(existing)
    if len(added) != expected:
        raise ValueError("regional extension did not fill the requested sample size")


def _format_group_stratum_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {group: dict(sorted(counts[group].items())) for group in GROUPS}


def _selected_stratum_counts(
    existing: Sequence[Mapping[str, Any]],
    added: Sequence[Mapping[str, Any]],
    stratum_of: StratumOf,
) -> dict[str, dict[str, int]]:
    counts = _existing_stratum_counts([*existing, *added], stratum_of)
    return counts


def _validate_existing_sample(existing: Sequence[Mapping[str, Any]], per_group: int) -> None:
    if _has_unknown_decision(existing):
        raise ValueError("existing sample has an unknown decision")
    for group in GROUPS:
        _validate_existing_group(existing, group, per_group)
    _validate_existing_identities(existing)


def _has_unknown_decision(existing: Sequence[Mapping[str, Any]]) -> bool:
    return any(row.get("decision") not in GROUPS for row in existing)


def _validate_existing_group(
    existing: Sequence[Mapping[str, Any]], group: str, per_group: int
) -> None:
    rows = _existing_group_rows(existing, group)
    _require_group_capacity(rows, group, per_group)
    _require_unique_group_cell_values(rows, group)


def _existing_group_rows(
    existing: Sequence[Mapping[str, Any]], group: str
) -> list[Mapping[str, Any]]:
    return [row for row in existing if row["decision"] == group]


def _require_group_capacity(rows: Sequence[Mapping[str, Any]], group: str, per_group: int) -> None:
    if len(rows) > per_group:
        raise ValueError(f"existing sample has more than {per_group} {group} rows")


def _require_unique_group_cell_values(rows: Sequence[Mapping[str, Any]], group: str) -> None:
    cells = [row["h3_cell"] for row in rows]
    if len(set(cells)) != len(cells):
        raise ValueError(f"existing sample has duplicate {group} cells")


def _validate_existing_identities(existing: Sequence[Mapping[str, Any]]) -> None:
    texts = [row["text_sha256"] for row in existing]
    polygons = [_polygon_key(row) for row in existing]
    if len(set(texts)) != len(texts) or len(set(polygons)) != len(polygons):
        raise ValueError("existing sample must have unique sentence hashes and polygons")


def _remove_existing_identities(
    rows: Sequence[Mapping[str, Any]], existing: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    existing_text = {row["text_sha256"] for row in existing}
    existing_polygons = {_polygon_key(row) for row in existing}
    return [row for row in rows if not _identity_is_existing(row, existing_text, existing_polygons)]


def _identity_is_existing(
    row: Mapping[str, Any],
    existing_text: set[str],
    existing_polygons: set[tuple[str, str, int]],
) -> bool:
    return row["text_sha256"] in existing_text or _polygon_key(row) in existing_polygons


def _extend_group(
    group: str,
    cells: Mapping[str, Sequence[Mapping[str, Any]]],
    existing: Sequence[Mapping[str, Any]],
    per_group: int,
    centre_of: CentreOf,
    seed: int,
) -> list[dict[str, Any]]:
    chosen, candidates, centres, nearest = _extension_state(
        group, cells, existing, per_group, centre_of
    )
    remaining = per_group - len(chosen)
    added: list[dict[str, Any]] = []
    while candidates and remaining:
        cell = _farthest_extension_cell(candidates, nearest, seed)
        candidates.remove(cell)
        remaining -= 1
        _update_nearest_extension_distances(candidates, nearest, centres, cell)
        added.append(_row_for_extension_cell(cells[cell][0], cell, centres[cell]))
    return added


def _extension_state(
    group: str,
    cells: Mapping[str, Sequence[Mapping[str, Any]]],
    existing: Sequence[Mapping[str, Any]],
    per_group: int,
    centre_of: CentreOf,
) -> tuple[list[str], list[str], dict[str, tuple[float, float]], dict[str, float]]:
    chosen = _chosen_extension_cells(group, existing)
    candidates = _available_extension_cells(cells, chosen)
    needed = per_group - len(chosen)
    _require_extension_capacity(candidates, needed, group)
    centres = _extension_centres(candidates, _chosen_extension_centres(group, existing), centre_of)
    nearest = _nearest_extension_distances(candidates, chosen, centres)
    return chosen, candidates, centres, nearest


def _chosen_extension_cells(group: str, existing: Sequence[Mapping[str, Any]]) -> list[str]:
    return [row["h3_cell"] for row in existing if row["decision"] == group]


def _chosen_extension_centres(
    group: str, existing: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[float, float]]:
    return {
        row["h3_cell"]: (row["cell_centre_lat"], row["cell_centre_lon"])
        for row in existing
        if row["decision"] == group
    }


def _available_extension_cells(cells: Mapping[str, Any], chosen: Sequence[str]) -> list[str]:
    taken = set(chosen)
    return [cell for cell in sorted(cells) if cell not in taken]


def _require_extension_capacity(candidates: Sequence[str], needed: int, group: str) -> None:
    if len(candidates) < needed:
        raise ValueError(f"need {needed} more {group} cells, found {len(candidates)}")


def _extension_centres(
    candidates: Sequence[str],
    chosen_centres: Mapping[str, tuple[float, float]],
    centre_of: CentreOf,
) -> dict[str, tuple[float, float]]:
    return {**chosen_centres, **{cell: centre_of(cell) for cell in candidates}}


def _nearest_extension_distances(
    candidates: Sequence[str],
    chosen: Sequence[str],
    centres: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    return {
        cell: min(
            (_distance_from_chosen(cell, other, centres) for other in chosen), default=math.inf
        )
        for cell in candidates
    }


def _distance_from_chosen(
    cell: str, chosen: str, centres: Mapping[str, tuple[float, float]]
) -> float:
    return _haversine_km(centres[cell], centres[chosen])


def _farthest_extension_cell(
    candidates: Sequence[str], nearest: Mapping[str, float], seed: int
) -> str:
    return max(candidates, key=lambda cell: (nearest[cell], _stable_key(seed, cell)))


def _update_nearest_extension_distances(
    candidates: Sequence[str],
    nearest: dict[str, float],
    centres: Mapping[str, tuple[float, float]],
    chosen: str,
) -> None:
    for cell in candidates:
        nearest[cell] = min(nearest[cell], _haversine_km(centres[cell], centres[chosen]))


def _row_for_extension_cell(
    row: Mapping[str, Any], cell: str, centre: tuple[float, float]
) -> dict[str, Any]:
    return {
        **row,
        "h3_cell": cell,
        "cell_centre_lat": centre[0],
        "cell_centre_lon": centre[1],
    }
