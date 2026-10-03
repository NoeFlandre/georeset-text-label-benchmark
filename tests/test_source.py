"""Pinned source discovery and projected-Parquet reader tests."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

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
from georeset_text_label_benchmark.source import DescriptionSource, HubFileStore


class MemoryStore:
    def __init__(self, names: dict[str, list[str]], payload: bytes = b"") -> None:
        self.names = names
        self.payload = payload
        self.opened_paths: list[str] = []
        self.opened_modes: list[str | None] = []

    def glob(self, pattern: str) -> list[str]:
        if "/labels/" in pattern:
            key = "labels"
        elif "/language-v1/data/" in pattern:
            key = "descriptions"
        else:
            key = "polygons"
        return [pattern.removesuffix("/*.parquet") + f"/{name}" for name in self.names[key]]

    def open(self, path: str, mode: str | None = "missing-mode") -> IO[bytes]:
        self.opened_paths.append(path)
        self.opened_modes.append(mode)
        local_path = Path(path)
        if local_path.is_file():
            assert mode is not None
            return local_path.open(mode)
        return io.BytesIO(self.payload)


def _snapshots() -> DatasetSnapshots:
    return DatasetSnapshots("owner/labels", "label-sha", "owner/eunis", "eunis-sha", "input-sha")


def _store(names: Sequence[str] = ("a.parquet", "b.parquet")) -> MemoryStore:
    files = list(names)
    return MemoryStore({"labels": files, "descriptions": files, "polygons": files})


def test_hub_file_store_opens_binary_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    mode = inspect.signature(HubFileStore.open).parameters["mode"]
    modes: list[str] = []

    class FakeFilesystem:
        def __init__(self, *, token: bool) -> None:
            assert token is False

        def open(self, path: str, mode: str) -> io.BytesIO:
            assert path == "manifest.json"
            modes.append(mode)
            return io.BytesIO(b"manifest")

    monkeypatch.setattr(source_module, "HfFileSystem", FakeFilesystem)
    store = HubFileStore()

    with store.open("manifest.json") as stream:
        assert stream.read() == b"manifest"

    assert mode.default == "rb"
    assert modes == ["rb"]


def test_projected_reader_default_batch_size_is_fixed() -> None:
    default = inspect.signature(DescriptionSource.read_rows).parameters["batch_size"].default

    assert default == 8_192


def test_discovery_aligns_three_source_files_by_filename() -> None:
    source = DescriptionSource(_store(), _snapshots(), expected_shards=2)

    partitions = source.partitions()
    indexes = source._partition_indexes()

    assert set(indexes) == {"labels", "descriptions", "polygons"}
    assert list(indexes["labels"]) == ["a.parquet", "b.parquet"]
    assert indexes["labels"]["a.parquet"] == (
        "datasets/owner/labels@label-sha/labels/language-v1/data/a.parquet"
    )
    assert indexes["descriptions"]["a.parquet"] == (
        "datasets/owner/eunis@eunis-sha/language-v1/data/a.parquet"
    )
    assert indexes["polygons"]["a.parquet"] == "datasets/owner/eunis@eunis-sha/data/a.parquet"
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
    with pytest.raises(
        CardinalityError,
        match=r"datasets/owner/eunis@eunis-sha/language-v1/data has 1 shards; expected 2",
    ):
        DescriptionSource(store, _snapshots(), expected_shards=2).partitions()


def test_discovery_rejects_different_filenames_even_when_counts_match() -> None:
    store = MemoryStore(
        {
            "labels": ["a.parquet", "b.parquet"],
            "descriptions": ["a.parquet", "c.parquet"],
            "polygons": ["a.parquet", "b.parquet"],
        }
    )
    with pytest.raises(
        CardinalityError,
        match=r"^source shard filenames do not match across all three snapshots$",
    ) as caught:
        DescriptionSource(store, _snapshots(), expected_shards=2).partitions()

    assert str(caught.value) == "source shard filenames do not match across all three snapshots"


def test_discovery_rejects_duplicate_file_names() -> None:
    store = MemoryStore(
        {
            "labels": ["a.parquet", "a.parquet"],
            "descriptions": ["a.parquet"],
            "polygons": ["a.parquet"],
        }
    )
    with pytest.raises(CardinalityError, match=r"labels has duplicate shard filename a\.parquet"):
        DescriptionSource(store, _snapshots(), expected_shards=None).partitions()


def test_discovery_rejects_missing_parquet_collection() -> None:
    store = MemoryStore({"labels": [], "descriptions": ["a.parquet"], "polygons": ["a.parquet"]})

    with pytest.raises(
        DataValidationError,
        match=r"no Parquet files found under datasets/owner/labels@label-sha/labels/language-v1/data",
    ):
        DescriptionSource(store, _snapshots(), expected_shards=None).partitions()


def test_read_rows_projects_requested_columns_in_small_batches(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"kept": [1, 2, 3], "ignored": ["x", "y", "z"]}), path)
    store = MemoryStore({"labels": [], "descriptions": [], "polygons": []})
    source = DescriptionSource(store, _snapshots())

    rows = list(source.read_rows(str(path), ["kept"], batch_size=1))

    assert rows == [{"kept": 1}, {"kept": 2}, {"kept": 3}]
    assert store.opened_modes == ["rb"]


def test_read_rows_forwards_the_projection_and_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryStore({"labels": [], "descriptions": [], "polygons": []}, payload=b"fake parquet")
    source = DescriptionSource(store, _snapshots())
    observed: list[dict[str, Any]] = []

    class FakeBatch:
        def to_pylist(self) -> list[dict[str, int]]:
            return [{"kept": 7}]

    class FakeParquet:
        schema_arrow = SimpleNamespace(names=["kept", "ignored"])

        def __init__(self, stream: IO[bytes]) -> None:
            assert stream.read() == b"fake parquet"

        def iter_batches(self, *, batch_size: int, columns: list[str]) -> list[FakeBatch]:
            observed.append({"batch_size": batch_size, "columns": columns})
            return [FakeBatch()]

    monkeypatch.setattr(source_module.pq, "ParquetFile", FakeParquet)

    assert list(source.read_rows("rows.parquet", ["kept"], batch_size=7)) == [{"kept": 7}]
    assert list(source.read_rows("rows.parquet", ["kept"])) == [{"kept": 7}]
    assert observed == [
        {"batch_size": 7, "columns": ["kept"]},
        {"batch_size": 8_192, "columns": ["kept"]},
    ]
    assert store.opened_modes == ["rb", "rb"]


def test_read_rows_rejects_missing_projected_columns(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"present": [1]}), path)
    source = DescriptionSource(
        MemoryStore({"labels": [], "descriptions": [], "polygons": []}), _snapshots()
    )

    with pytest.raises(
        DataValidationError,
        match=re.escape(f"{path} is missing required columns: absent"),
    ):
        list(source.read_rows(str(path), ["absent"]))

    with pytest.raises(
        DataValidationError,
        match=re.escape(f"{path} is missing required columns: alpha, zeta"),
    ) as caught:
        DescriptionSource._require_columns(str(path), ["zeta", "alpha"], ["present"])

    assert str(caught.value) == f"{path} is missing required columns: alpha, zeta"


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
    with store.open("manifest.json", "rb") as stream:
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

    assert result == {
        "source_revision": "input-sha",
        "eunis_reference_version": "EEA maps v1",
        "eunis_reference_asset_count": 1,
        "eunis_manifest_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
    }
    assert store.opened_paths == ["datasets/owner/eunis@eunis-sha/eunis/manifest.json"]
    assert store.opened_modes == ["rb"]


def test_eunis_manifest_rejects_other_source_revision() -> None:
    manifest = {"source_revision": "not-the-pinned-input", "reference": {"source_version": "v1"}}
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )

    with pytest.raises(
        ProvenanceError,
        match=r"^EUNIS manifest source_revision differs from the pinned input$",
    ) as caught:
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()

    assert str(caught.value) == "EUNIS manifest source_revision differs from the pinned input"


def test_eunis_manifest_rejects_non_object_root() -> None:
    store = MemoryStore({"labels": [], "descriptions": [], "polygons": []}, b"[]")

    with pytest.raises(
        DataValidationError,
        match=r"^EUNIS manifest root must be a JSON object$",
    ) as caught:
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()

    assert str(caught.value) == "EUNIS manifest root must be a JSON object"


def test_eunis_manifest_rejects_missing_asset_list() -> None:
    manifest = {"source_revision": "input-sha", "reference": {"source_version": "v1"}}
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )

    with pytest.raises(
        DataValidationError,
        match=r"^EUNIS manifest reference must list its source assets$",
    ) as caught:
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()

    assert str(caught.value) == "EUNIS manifest reference must list its source assets"


def test_eunis_manifest_rejects_empty_source_version() -> None:
    manifest = {
        "source_revision": "input-sha",
        "reference": {"source_version": " ", "assets": [{"code": "T11"}]},
    }
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )

    with pytest.raises(
        DataValidationError,
        match=r"^EUNIS manifest is missing its reference source version$",
    ) as caught:
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()

    assert str(caught.value) == "EUNIS manifest is missing its reference source version"


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        ([], "EUNIS manifest is missing its reference source version"),
        (
            {"source_version": 1, "assets": []},
            "EUNIS manifest is missing its reference source version",
        ),
        (
            {"source_version": "v1", "assets": []},
            "EUNIS manifest reference must list its source assets",
        ),
    ],
)
def test_eunis_manifest_rejects_invalid_reference_sections(reference, error: str) -> None:
    manifest = {"source_revision": "input-sha", "reference": reference}
    store = MemoryStore(
        {"labels": [], "descriptions": [], "polygons": []}, json.dumps(manifest).encode()
    )

    with pytest.raises(DataValidationError, match=f"^{re.escape(error)}$") as caught:
        DescriptionSource(store, _snapshots(), expected_shards=None).eunis_manifest()

    assert str(caught.value) == error
