"""Tests for extending the published geographic pilot sample to more rows per group."""

from __future__ import annotations

import pytest

from georeset_text_label_benchmark.pilot.geo_sampling import (
    extend_geographic_sample,
    h3_cell_of,
    h3_centre_of,
    select_geographic_sample,
)


def _row(index: int, decision: str, lat: float, lon: float) -> dict[str, object]:
    return {
        "source_pbf": "test-latest",
        "osm_type": "way",
        "osm_id": index,
        "description_identity": f"desc-{index}",
        "tag_key": "description",
        "sentence_index": 0,
        "text_sha256": f"text-{index}",
        "sentence": f"sentence {index}",
        "language_code": "eng",
        "eunis_code": "Q11",
        "decision": decision,
        "bbox_min_x": lon - 0.01,
        "bbox_max_x": lon + 0.01,
        "bbox_min_y": lat - 0.01,
        "bbox_max_y": lat + 0.01,
    }


def _grid_rows() -> list[dict[str, object]]:
    rows = []
    index = 0
    for decision in ("yes", "no"):
        for i in range(12):
            for j in range(12):
                index += 1
                lat = 40.0 + 1.5 * i + (0.7 if decision == "no" else 0.0)
                lon = -4.0 + 2.0 * j + (0.9 if decision == "no" else 0.0)
                rows.append(_row(index, decision, lat, lon))
    return rows


def test_extension_keeps_base_rows_and_adds_distinct_cells_per_group() -> None:
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
    selected = [*base, *added]
    for group in ("yes", "no"):
        group_cells = [row["h3_cell"] for row in selected if row["decision"] == group]
        assert len(group_cells) == 12
        assert len(set(group_cells)) == 12
    base_ids = {row["text_sha256"] for row in base}
    assert base_ids.issubset({row["text_sha256"] for row in selected})
    assert len({row["text_sha256"] for row in selected}) == len(selected)


def test_extension_rejects_request_below_existing_count() -> None:
    rows = _grid_rows()
    base = select_geographic_sample(
        rows, cell_of=h3_cell_of, centre_of=h3_centre_of, per_group=5, seed=42
    )
    with pytest.raises(ValueError, match="more than 4"):
        extend_geographic_sample(
            rows,
            cell_of=h3_cell_of,
            centre_of=h3_centre_of,
            existing=base,
            per_group=4,
            seed=42,
        )
