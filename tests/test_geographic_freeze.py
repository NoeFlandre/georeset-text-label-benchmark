"""Freeze and read the geographic yes/no sample from a synthetic labelled pool."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset_text_label_benchmark.output_schema import LABELLED_SCHEMA
from georeset_text_label_benchmark.pilot import cli
from georeset_text_label_benchmark.pilot.geo_sampling import h3_cell_of, h3_centre_of
from georeset_text_label_benchmark.pilot.runner import freeze_sample, read_frozen_pilot_inputs


def _candidate_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pilot_data/eunis_candidate_labels.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pinned EUNIS candidate table is missing")


def _candidates() -> list[dict[str, str]]:
    with _candidate_file().open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pool(directory: Path, extra: Sequence[dict[str, Any]] = ()) -> tuple[Path, Path]:
    """Every candidate code gets one yes and one no sentence at a random place on Earth."""
    rng = random.Random(7)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(_candidates()):
        for offset, decision in enumerate(("yes", "no")):
            lat = rng.uniform(-60.0, 70.0)
            lon = rng.uniform(-170.0, 170.0)
            sentence = f"Pool sentence {index} {decision}."
            rows.append(
                {
                    "source_pbf": "test-latest.osm.pbf",
                    "osm_type": "way",
                    "osm_id": index * 2 + offset + 1,
                    "osm_url": None,
                    "description_identity": f"description-{index}-{decision}",
                    "tag_key": "description",
                    "sentence_index": 0,
                    "sentence": sentence,
                    "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
                    "language_code": "eng",
                    "eunis_code": candidate["eunis_code"],
                    "eunis_name": candidate["eunis_name"],
                    "eunis_overlap_percentage": 80.0,
                    "eunis_source_version": "maps-v1",
                    "decision": decision,
                    "bbox_min_x": lon - 0.1,
                    "bbox_min_y": lat - 0.1,
                    "bbox_max_x": lon + 0.1,
                    "bbox_max_y": lat + 0.1,
                }
            )
    rows.extend(extra)
    rows = [{key: row.get(key) for key in LABELLED_SCHEMA.names} for row in rows]
    labelled = directory / "labelled-eunis.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=LABELLED_SCHEMA), labelled)
    manifest = directory / "pipeline-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "computation_commit": "a" * 40,
                "validation_commit": "b" * 40,
                "input_snapshots": {"shared_input_revision": "shared-revision"},
                "artifact_sha256": {"labelled-eunis.parquet": _sha256(labelled)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return labelled, manifest


@pytest.fixture
def pool(tmp_path: Path) -> tuple[Path, Path, Path]:
    labelled, manifest = _write_pool(tmp_path)
    candidate = _candidate_file()
    return labelled, manifest, candidate


def _freeze(pool: tuple[Path, Path, Path], output: Path) -> dict[str, Any]:
    labelled, manifest, candidate = pool
    return freeze_sample(labelled, candidate, manifest, output)


def test_freeze_writes_fifty_yes_and_fifty_no_sentences_on_distinct_cells(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)

    sample = json.loads((output / "frozen_sample.json").read_text(encoding="utf-8"))
    rows = sample["selected_rows"]
    decisions = [row["decision"] for row in rows]
    cells = [row["h3_cell"] for row in rows]
    assert len(rows) == 100
    assert decisions.count("yes") == decisions.count("no") == 50
    assert len(set(cells)) == 100
    assert all(row["sample_id"] and row["text_sha256"] for row in rows)
    assert sample["selection"]["decision_counts"] == {"yes": 50, "no": 50}
    assert sample["selection"]["h3_resolution"] == 3


def test_freeze_copies_the_candidate_table_and_records_output_checksums(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)

    assert _sha256(output / "candidate_labels.csv") == _sha256(pool[2])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs_sha256"] == {
        "frozen_sample.json": _sha256(output / "frozen_sample.json"),
        "candidate_labels.csv": _sha256(output / "candidate_labels.csv"),
    }
    assert manifest["source_sha256"]["labelled-eunis.parquet"] == _sha256(pool[0])
    assert manifest["source_sha256"]["pipeline-manifest.json"] == _sha256(pool[1])


def test_freeze_rejects_a_labelled_pool_that_differs_from_the_pipeline_manifest(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    labelled, _manifest, _candidate = pool
    labelled.write_bytes(labelled.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=r"labelled-eunis\.parquet SHA-256"):
        _freeze(pool, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_freeze_rejects_a_candidate_table_that_is_not_the_pinned_one(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    changed = tmp_path / "changed.csv"
    shutil.copyfile(pool[2], changed)
    changed.write_bytes(changed.read_bytes() + b"\n")
    labelled, manifest, _candidate = pool

    with pytest.raises(ValueError, match=r"candidate CSV SHA-256"):
        freeze_sample(labelled, changed, manifest, tmp_path / "run")


def test_freeze_refuses_an_existing_output_directory(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        _freeze(pool, output)


def test_freeze_is_reproducible_for_the_same_pool(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _freeze(pool, tmp_path / "first")
    _freeze(pool, tmp_path / "second")

    assert _sha256(tmp_path / "first" / "frozen_sample.json") == _sha256(
        tmp_path / "second" / "frozen_sample.json"
    )


def test_reader_accepts_a_fresh_freeze(pool: tuple[Path, Path, Path], tmp_path: Path) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)

    _sample, rows, candidates = read_frozen_pilot_inputs(output)

    assert len(rows) == 100
    assert len(candidates) == 158


def test_reader_rejects_a_sample_that_is_not_fifty_fifty(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)
    path = output / "frozen_sample.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    first_yes = next(row for row in sample["selected_rows"] if row["decision"] == "yes")
    first_yes["decision"] = "no"
    path.write_text(json.dumps(sample), encoding="utf-8")

    with pytest.raises(ValueError, match=r"must have 50 yes and 50 no"):
        read_frozen_pilot_inputs(output)


def test_reader_rejects_a_cell_that_does_not_match_its_centre(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)
    path = output / "frozen_sample.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    row = sample["selected_rows"][0]
    lat, lon = row["cell_centre_lat"], row["cell_centre_lon"]
    row["h3_cell"] = (
        h3_cell_of(lat + 20.0, lon)
        if h3_cell_of(lat + 20.0, lon) != row["h3_cell"]
        else h3_cell_of(lat - 20.0, lon)
    )
    path.write_text(json.dumps(sample), encoding="utf-8")

    with pytest.raises(ValueError, match="H3 cell does not match its centre"):
        read_frozen_pilot_inputs(output)


def test_cli_freeze_takes_the_labelled_pool_and_pipeline_manifest(tmp_path: Path) -> None:
    parser = cli._parser()
    args = parser.parse_args(
        [
            "freeze",
            "--labelled-parquet",
            str(tmp_path / "labelled-eunis.parquet"),
            "--pipeline-manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert args.labelled_parquet == tmp_path / "labelled-eunis.parquet"
    assert args.pipeline_manifest == tmp_path / "manifest.json"
    assert args.per_group == 50
    assert args.seed == 42
    assert h3_centre_of(h3_cell_of(0.0, 0.0)) is not None


def test_cli_help_lists_the_geographic_freeze_and_dspark_commands() -> None:
    assert cli._parser().format_help() == (
        "usage: georeset-pilot [-h] {freeze,run-dspark,run-dspark-smoke} ...\n\n"
        "positional arguments:\n"
        "  {freeze,run-dspark,run-dspark-smoke}\n"
        "    freeze              freeze the geographic yes/no sample\n"
        "    run-dspark          predict one EUNIS code per row with pinned LFM2.5 +\n"
        "                        DSpark\n"
        "    run-dspark-smoke    run a bounded eight-row LFM2.5 + DSpark readiness\n"
        "                        smoke\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
    )


def test_reader_rejects_a_sample_that_does_not_name_the_labelled_pool(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)
    path = output / "frozen_sample.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    sample["source"]["file"] = "overlap.parquet"
    path.write_text(json.dumps(sample), encoding="utf-8")

    with pytest.raises(ValueError, match=r"not from the labelled EUNIS pool"):
        read_frozen_pilot_inputs(output)


def test_reader_rejects_a_sample_without_pipeline_checksum(
    pool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _freeze(pool, output)
    path = output / "frozen_sample.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    del sample["source"]["pipeline_manifest_sha256"]
    path.write_text(json.dumps(sample), encoding="utf-8")

    with pytest.raises(ValueError, match=r"source checksums are missing"):
        read_frozen_pilot_inputs(output)


def test_freeze_leaves_out_sentences_outside_the_candidate_vocabulary(tmp_path: Path) -> None:
    sentence = "Snow-bed sentence."
    outside = {
        "source_pbf": "austria-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": 900001,
        "description_identity": "description-outside",
        "tag_key": "description",
        "sentence_index": 0,
        "sentence": sentence,
        "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
        "language_code": "eng",
        "eunis_code": "R41",
        "eunis_name": "Snow-bed vegetation",
        "eunis_overlap_percentage": 80.0,
        "eunis_source_version": "maps-v1",
        "decision": "no",
        "bbox_min_x": 10.0,
        "bbox_min_y": 47.0,
        "bbox_max_x": 10.2,
        "bbox_max_y": 47.2,
    }
    labelled, manifest = _write_pool(tmp_path, extra=[outside])
    output = tmp_path / "run"

    freeze_sample(labelled, _candidate_file(), manifest, output)

    sample = json.loads((output / "frozen_sample.json").read_text(encoding="utf-8"))
    assert all(row["eunis_code"] != "R41" for row in sample["selected_rows"])
    assert sample["source_coverage"]["rows_outside_candidate_vocabulary"] == 1
    assert sample["selection"]["decision_counts"] == {"yes": 50, "no": 50}
