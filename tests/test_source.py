"""Pinned source discovery and projected-Parquet reader tests."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import IO

import pyarrow as pa
import pytest
from pyarrow import parquet as pq

from georeset_text_label_benchmark import source as source_module
from georeset_text_label_benchmark.config import DatasetSnapshots
from georeset_text_label_benchmark.errors import (
    CardinalityError,
    DataValidationError,
    ProvenanceError,
)
from georeset_text_label_benchmark.source import DescriptionSource


class MemoryStore:
    def __init__(self, names: dict[str, list[str]], payload: bytes = b"") -> None:
        self.names = names
        self.payload = payload

    def glob(self, pattern: str) -> list[str]:
        if "/labels/" in pattern:
            key = "labels"
        elif "/language-v1/data/" in pattern:
            key = "descriptions"
        else:
            key = "polygons"
        return [pattern.removesuffix("/*.parquet") + f"/{name}" for name in self.names[key]]

    def open(self, path: str, mode: str = "rb") -> IO[bytes]:
        local_path = Path(path)
        if local_path.is_file():
            return local_path.open(mode)
        return io.BytesIO(self.payload)


def _snapshots() -> DatasetSnapshots:
    return DatasetSnapshots("owner/labels", "label-sha", "owner/eunis", "eunis-sha", "input-sha")


def _store(names: Sequence[str] = ("a.parquet", "b.parquet")) -> MemoryStore:
    files = list(names)
    return MemoryStore({"labels": files, "descriptions": files, "polygons": files})


def test_discovery_aligns_three_source_files_by_filename() -> None:
    source = DescriptionSource(_store(), _snapshots(), expected_shards=2)

    partitions = source.partitions()

    assert [part.name for part in partitions] == ["a.parquet", "b.parquet"]
    assert partitions[0].labels_path.endswith("/labels/language-v1/data/a.parquet")
    assert partitions[0].descriptions_path.endswith("/language-v1/data/a.parquet")
    assert partitions[0].polygons_path.endswith("/data/a.parquet")


def test_discovery_rejects_missing_or_extra_regional_shards() -> None:
    store = MemoryStore(
        {
            "labels": ["a.parquet", "b.parquet"],
            "descriptions": ["a.parquet"],
            "polygons": ["a.parquet", "b.parquet"],
        }
    )
    with pytest.raises(CardinalityError, match="has 1 shards; expected 2"):
        DescriptionSource(store, _snapshots(), expected_shards=2).partitions()


def test_discovery_rejects_different_filenames_even_when_counts_match() -> None:
    store = MemoryStore(
        {
            "labels": ["a.parquet", "b.parquet"],
            "descriptions": ["a.parquet", "c.parquet"],
            "polygons": ["a.parquet", "b.parquet"],
        }
    )
    with pytest.raises(CardinalityError, match="filenames do not match"):
        DescriptionSource(store, _snapshots(), expected_shards=2).partitions()


def test_discovery_rejects_duplicate_file_names() -> None:
    store = MemoryStore(
        {
            "labels": ["a.parquet", "a.parquet"],
            "descriptions": ["a.parquet"],
            "polygons": ["a.parquet"],
        }
    )
    with pytest.raises(CardinalityError, match="duplicate shard filename"):
        DescriptionSource(store, _snapshots(), expected_shards=None).partitions()


def test_discovery_rejects_missing_parquet_collection() -> None:
    store = MemoryStore({"labels": [], "descriptions": ["a.parquet"], "polygons": ["a.parquet"]})

    with pytest.raises(DataValidationError, match="no Parquet files"):
        DescriptionSource(store, _snapshots(), expected_shards=None).partitions()


def test_read_rows_projects_requested_columns_in_small_batches(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"kept": [1, 2, 3], "ignored": ["x", "y", "z"]}), path)
    source = DescriptionSource(
        MemoryStore({"labels": [], "descriptions": [], "polygons": []}), _snapshots()
    )

    rows = list(source.read_rows(str(path), ["kept"], batch_size=1))

    assert rows == [{"kept": 1}, {"kept": 2}, {"kept": 3}]


def test_read_rows_rejects_missing_projected_columns(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"present": [1]}), path)
    source = DescriptionSource(
        MemoryStore({"labels": [], "descriptions": [], "polygons": []}), _snapshots()
    )

    with pytest.raises(DataValidationError, match="missing required columns: absent"):
        list(source.read_rows(str(path), ["absent"]))


def test_default_source_wraps_public_hub_filesystem(monkeypatch) -> None:
    class FakeFilesystem:
        def __init__(self, *, token: bool) -> None:
            assert token is False

        def glob(self, pattern: str) -> list[str]:
            return [pattern]

        def open(self, path: str, mode: str) -> io.BytesIO:
            assert mode == "rb"
            return io.BytesIO(path.encode())

    monkeypatch.setattr(source_module, "HfFileSystem", FakeFilesystem)
    store = DescriptionSource.from_hub()._store

    assert store.glob("path/*.parquet") == ["path/*.parquet"]
    with store.open("manifest.json") as stream:
        assert stream.read() == b"manifest.json"


def test_eunis_manifest_preserves_source_version_and_hash() -> None:
    manifest = {
        "source_revision": "input-sha",
        "reference": {"source_version": "EEA maps v1", "assets": [{"code": "T11"}]},
    }
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )
    source = DescriptionSource(store, _snapshots(), expected_shards=None)

    result = source.eunis_manifest()

    assert result["source_revision"] == "input-sha"
    assert result["eunis_reference_version"] == "EEA maps v1"
    assert result["eunis_reference_asset_count"] == 1
    assert len(result["eunis_manifest_sha256"]) == 64


def test_eunis_manifest_rejects_other_source_revision() -> None:
    manifest = {"source_revision": "not-the-pinned-input", "reference": {"source_version": "v1"}}
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )

    with pytest.raises(ProvenanceError, match="source_revision"):
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()


def test_eunis_manifest_rejects_non_object_root() -> None:
    store = MemoryStore({"labels": [], "descriptions": [], "polygons": []}, b"[]")

    with pytest.raises(DataValidationError, match="root must be a JSON object"):
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()


def test_eunis_manifest_rejects_missing_asset_list() -> None:
    manifest = {"source_revision": "input-sha", "reference": {"source_version": "v1"}}
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )

    with pytest.raises(DataValidationError, match="source assets"):
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()


def test_eunis_manifest_rejects_empty_source_version() -> None:
    manifest = {
        "source_revision": "input-sha",
        "reference": {"source_version": " ", "assets": [{"code": "T11"}]},
    }
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )

    with pytest.raises(DataValidationError, match="source version"):
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()


@pytest.mark.parametrize(
    "reference",
    [[], {"source_version": 1, "assets": []}, {"source_version": "v1", "assets": []}],
)
def test_eunis_manifest_rejects_invalid_reference_sections(reference) -> None:
    manifest = {"source_revision": "input-sha", "reference": reference}
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )

    with pytest.raises(DataValidationError, match=r"source version|source assets"):
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()
