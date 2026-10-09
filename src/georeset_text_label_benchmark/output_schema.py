"""Stable, compact Arrow schemas for retained text/polygon occurrences."""

from __future__ import annotations

import pyarrow as pa

OVERLAP_SCHEMA = pa.schema(
    [
        pa.field("source_pbf", pa.string(), nullable=False),
        pa.field("osm_type", pa.string(), nullable=False),
        pa.field("osm_id", pa.int64(), nullable=False),
        pa.field("description_identity", pa.string(), nullable=False),
        pa.field("tag_key", pa.string(), nullable=False),
        pa.field("sentence_index", pa.int32(), nullable=False),
        pa.field("sentence", pa.string(), nullable=False),
        pa.field("text_sha256", pa.string(), nullable=False),
        pa.field("language_code", pa.string()),
        pa.field("eunis_code", pa.string(), nullable=False),
        pa.field("eunis_name", pa.string(), nullable=False),
        pa.field("eunis_overlap_percentage", pa.float64(), nullable=False),
        pa.field("eunis_source_version", pa.string(), nullable=False),
    ]
)

LABELLED_SCHEMA = pa.schema(
    [
        *OVERLAP_SCHEMA,
        pa.field("decision", pa.string(), nullable=False),
        pa.field("bbox_min_x", pa.float64(), nullable=False),
        pa.field("bbox_min_y", pa.float64(), nullable=False),
        pa.field("bbox_max_x", pa.float64(), nullable=False),
        pa.field("bbox_max_y", pa.float64(), nullable=False),
    ]
)
