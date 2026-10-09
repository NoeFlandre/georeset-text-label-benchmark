"""Pinned Description source adapter with projected, shard-bounded Parquet reads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath
from typing import IO, Any, Protocol

from huggingface_hub import HfFileSystem
from pyarrow import parquet as pq

from georeset_text_label_benchmark.config import (
    DEFAULT_SNAPSHOTS,
    EXPECTED_SHARDS,
    DatasetSnapshots,
)
from georeset_text_label_benchmark.errors import (
    CardinalityError,
    DataValidationError,
    ProvenanceError,
)
from georeset_text_label_benchmark.models import SourcePartition

POLYGON_COLUMNS = (
    "source_pbf",
    "osm_type",
    "osm_id",
    "eunis_code",
    "eunis_name",
    "eunis_overlap_percentage",
    "eunis_source_version",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_max_x",
    "bbox_max_y",
)
DESCRIPTION_COLUMNS = (
    "description_identity",
    "source_pbf",
    "osm_type",
    "osm_id",
    "tag_key",
    "language_code",
    "sentence_count",
    "sentences",
)
LABEL_COLUMNS = (
    "description_identity",
    "tag_key",
    "sentence_index",
    "text_sha256",
    "decision",
    "language",
    "input_revision",
)


class FileStore(Protocol):
    """Read-only path listing and binary stream interface for local or HF files."""

    def glob(self, pattern: str) -> list[str]: ...

    def open(self, path: str, mode: str) -> IO[bytes]: ...


class HubFileStore:
    """Public Hub reader; no write scopes or shared Hub cache are required."""

    def __init__(self) -> None:
        self._filesystem = HfFileSystem(token=False)

    def glob(self, pattern: str) -> list[str]:
        return self._filesystem.glob(pattern)

    def open(self, path: str, mode: str = "rb") -> IO[bytes]:
        return self._filesystem.open(path, mode)


class DescriptionSource:
    """Normalize the three pinned Description collections into aligned shards."""

    def __init__(
        self,
        store: FileStore,
        snapshots: DatasetSnapshots = DEFAULT_SNAPSHOTS,
        expected_shards: int | None = EXPECTED_SHARDS,
    ) -> None:
        self._store = store
        self.snapshots = snapshots
        self._expected_shards = expected_shards

    @classmethod
    def from_hub(cls) -> DescriptionSource:
        """Create the default read-only adapter for the verified public snapshots."""
        return cls(HubFileStore())

    def partitions(self) -> list[SourcePartition]:
        """Return filename-aligned labels, description values, and polygon shards."""
        indexes = self._partition_indexes()
        names = set(indexes["labels"])
        if names != set(indexes["descriptions"]) or names != set(indexes["polygons"]):
            raise CardinalityError("source shard filenames do not match across all three snapshots")
        return [
            SourcePartition(
                name=filename,
                labels_path=indexes["labels"][filename],
                descriptions_path=indexes["descriptions"][filename],
                polygons_path=indexes["polygons"][filename],
            )
            for filename in sorted(names)
        ]

    def _partition_indexes(self) -> dict[str, dict[str, str]]:
        return {
            name: self._by_filename(name, self._files(root, prefix))
            for name, (root, prefix) in self._roots().items()
        }

    def read_rows(
        self, path: str, columns: Sequence[str], batch_size: int = 8_192
    ) -> Iterable[dict[str, Any]]:
        """Stream selected Parquet columns from one shard in bounded row batches."""
        with self._store.open(path, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            self._require_columns(path, columns, parquet.schema_arrow.names)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
                yield from batch.to_pylist()

    def eunis_manifest(self) -> dict[str, Any]:
        """Read and pin the source EUNIS manifest without downloading polygon data."""
        root = self._repo_root(self.snapshots.eunis_repo, self.snapshots.eunis_revision)
        path = f"{root}/eunis/manifest.json"
        with self._store.open(path, "rb") as stream:
            raw = stream.read()
        manifest = json.loads(raw)
        _validate_manifest_root(manifest)
        _validate_manifest_source(manifest, self.snapshots.input_revision)
        reference = _require_eunis_reference(manifest.get("reference"))
        assets = _require_reference_assets(reference)
        return {
            "source_revision": manifest["source_revision"],
            "eunis_reference_version": reference["source_version"],
            "eunis_reference_asset_count": len(assets),
            "eunis_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def _roots(self) -> dict[str, tuple[str, str]]:
        return {
            "labels": (
                self._repo_root(self.snapshots.labels_repo, self.snapshots.labels_revision),
                "labels/language-v1/data",
            ),
            "descriptions": (
                self._repo_root(self.snapshots.eunis_repo, self.snapshots.eunis_revision),
                "language-v1/data",
            ),
            "polygons": (
                self._repo_root(self.snapshots.eunis_repo, self.snapshots.eunis_revision),
                "data",
            ),
        }

    @staticmethod
    def _repo_root(repo: str, revision: str) -> str:
        return f"datasets/{repo}@{revision}"

    def _files(self, root: str, prefix: str) -> list[str]:
        files = sorted(self._store.glob(f"{root}/{prefix}/*.parquet"))
        if not files:
            raise DataValidationError(f"no Parquet files found under {root}/{prefix}")
        if self._expected_shards is not None and len(files) != self._expected_shards:
            raise CardinalityError(
                f"{root}/{prefix} has {len(files)} shards; expected {self._expected_shards}"
            )
        return files

    @staticmethod
    def _by_filename(name: str, paths: list[str]) -> dict[str, str]:
        index: dict[str, str] = {}
        for path in paths:
            filename = PurePosixPath(path).name
            if filename in index:
                raise CardinalityError(f"{name} has duplicate shard filename {filename}")
            index[filename] = path
        return index

    @staticmethod
    def _require_columns(path: str, requested: Sequence[str], available: list[str]) -> None:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise DataValidationError(f"{path} is missing required columns: {', '.join(missing)}")


def _validate_manifest_root(value: Any) -> None:
    if not isinstance(value, dict):
        raise DataValidationError("EUNIS manifest root must be a JSON object")


def _validate_manifest_source(manifest: dict[str, Any], expected: str) -> None:
    if manifest.get("source_revision") != expected:
        raise ProvenanceError("EUNIS manifest source_revision differs from the pinned input")


def _require_eunis_reference(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataValidationError("EUNIS manifest is missing its reference source version")
    source_version = value.get("source_version")
    if not isinstance(source_version, str):
        raise DataValidationError("EUNIS manifest is missing its reference source version")
    if not source_version.strip():
        raise DataValidationError("EUNIS manifest is missing its reference source version")
    return value


def _require_reference_assets(reference: dict[str, Any]) -> list[Any]:
    assets = reference.get("assets")
    if not isinstance(assets, list):
        raise DataValidationError("EUNIS manifest reference must list its source assets")
    if not assets:
        raise DataValidationError("EUNIS manifest reference must list its source assets")
    return assets
