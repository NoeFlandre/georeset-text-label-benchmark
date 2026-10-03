"""Small domain records shared by source adapters and join orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourcePartition:
    """Three aligned Parquet shards for one source PBF extract."""

    name: str
    labels_path: str
    descriptions_path: str
    polygons_path: str


@dataclass(slots=True)
class GlobalKeys:
    """Occurrence keys seen across partitions; hashes are never join keys."""

    polygon_keys: set[tuple[str, int]] = field(default_factory=set)
    described_polygon_keys: set[tuple[str, int]] = field(default_factory=set)
    description_keys: set[tuple[str, str]] = field(default_factory=set)
    sentence_keys: set[tuple[str, str, int]] = field(default_factory=set)


@dataclass(slots=True)
class AuditCounts:
    """Streaming counts for source stages and the retained overlap rows."""

    polygon_rows: int = 0
    assigned_polygon_rows: int = 0
    description_tag_rows: int = 0
    label_rows: int = 0
    sentence_rows: int = 0
    skipped_unsplit_rows: int = 0
    retained_rows: int = 0
    yes_without_eunis_rows: int = 0
    decisions: Counter[str] = field(default_factory=Counter)
    decision_assignment: Counter[tuple[str, str]] = field(default_factory=Counter)
    overlap_by_eunis: Counter[str] = field(default_factory=Counter)
    overlap_by_tag: Counter[str] = field(default_factory=Counter)
    overlap_by_language: Counter[str] = field(default_factory=Counter)
    unique_sentence_hashes: set[str] = field(default_factory=set)

    def merge(self, other: AuditCounts) -> None:
        """Add one partition's counts to the run totals."""
        self.polygon_rows += other.polygon_rows
        self.assigned_polygon_rows += other.assigned_polygon_rows
        self.description_tag_rows += other.description_tag_rows
        self.label_rows += other.label_rows
        self.sentence_rows += other.sentence_rows
        self.skipped_unsplit_rows += other.skipped_unsplit_rows
        self.retained_rows += other.retained_rows
        self.yes_without_eunis_rows += other.yes_without_eunis_rows
        self.decisions.update(other.decisions)
        self.decision_assignment.update(other.decision_assignment)
        self.overlap_by_eunis.update(other.overlap_by_eunis)
        self.overlap_by_tag.update(other.overlap_by_tag)
        self.overlap_by_language.update(other.overlap_by_language)
        self.unique_sentence_hashes.update(other.unique_sentence_hashes)


@dataclass(slots=True)
class PartitionResult:
    """Eligible overlap rows and stage counts from one bounded partition."""

    overlap_rows: list[dict[str, Any]]
    audit: AuditCounts
