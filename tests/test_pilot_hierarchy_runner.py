"""Retrospective scoring publishes verified derived files without touching its source run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset_text_label_benchmark.pilot import cli
from georeset_text_label_benchmark.pilot.hierarchy_runner import score_hierarchical_run
from georeset_text_label_benchmark.pilot.metrics import single_label_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _selected_rows(sample_size: int) -> list[dict[str, Any]]:
    if sample_size == 2:
        return [
            {
                "sample_id": "sample-yes",
                "decision": "yes",
                "eunis_code": "A11",
                "eunis_source_version": "EEA map v1",
            },
            {
                "sample_id": "sample-no",
                "decision": "no",
                "eunis_code": "A12",
                "eunis_source_version": "EEA map v1",
            },
        ]
    if sample_size % 2:
        raise ValueError("large hierarchy fixtures require an even sample size")
    return [_large_sample_row(index, sample_size) for index in range(sample_size)]


def _large_sample_row(index: int, sample_size: int) -> dict[str, Any]:
    decision = "yes" if index < sample_size // 2 else "no"
    code = "B11" if index % 100 == 0 else "A11" if index % 2 == 0 else "A12"
    return {
        "sample_id": f"sample-{index:04d}",
        "decision": decision,
        "eunis_code": code,
        "eunis_source_version": "EEA map v1",
    }


def _prediction_rows(sample_size: int, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if sample_size == 2:
        return [
            {**selected[0], "parsed_eunis_code": "A11", "parse_status": "valid"},
            {**selected[1], "parsed_eunis_code": None, "parse_status": "truncated"},
        ]
    return [_large_prediction_row(index, row) for index, row in enumerate(selected)]


def _large_prediction_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    status = {1: "truncated", 2: "invalid"}.get(index % 10, "valid")
    return {
        **row,
        "parsed_eunis_code": row["eunis_code"] if status == "valid" else None,
        "parse_status": status,
    }


def _fixtures(tmp_path: Path, sample_size: int = 2) -> dict[str, Path]:
    frozen_dir = tmp_path / "frozen"
    run_dir = tmp_path / "run"
    frozen_dir.mkdir()
    run_dir.mkdir()
    candidate_path = frozen_dir / "candidate_labels.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "eunis_code",
                "classification_release",
                "classification_source",
                "source_sha256",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        candidate_rows = [
            (
                "A11",
                "EUNIS-test-terrestrial-v1",
                "https://example.test/terrestrial",
                "a" * 64,
            ),
            (
                "A12",
                "EUNIS-test-terrestrial-v1",
                "https://example.test/terrestrial",
                "a" * 64,
            ),
            ("B11", "EUNIS-test-marine-v1", "https://example.test/marine", "b" * 64),
        ]
        for code, release, source, source_hash in candidate_rows:
            writer.writerow(
                {
                    "eunis_code": code,
                    "classification_release": release,
                    "classification_source": source,
                    "source_sha256": source_hash,
                }
            )
    candidate_hash = _sha256(candidate_path)
    selected = _selected_rows(sample_size)
    ids = [row["sample_id"] for row in selected]
    selection = {"sample_size": sample_size, "sample_ids_sha256": _sha256_json(ids), "seed": 42}
    sample = {
        "format_version": 2,
        "candidate_labels": {"codes": [row[0] for row in candidate_rows], "sha256": candidate_hash},
        "selection": selection,
        "selected_rows": selected,
    }
    sample_path = frozen_dir / "frozen_sample.json"
    _write_json(sample_path, sample)
    frozen_manifest_path = frozen_dir / "manifest.json"
    _write_json(
        frozen_manifest_path,
        {
            "sample_ids_sha256": selection["sample_ids_sha256"],
            "outputs_sha256": {
                "frozen_sample.json": _sha256(sample_path),
                "candidate_labels.csv": candidate_hash,
            },
        },
    )

    predictions = _prediction_rows(sample_size, selected)
    predictions_path = run_dir / "dspark_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(predictions), predictions_path, compression="zstd")
    metrics_path = run_dir / "dspark_metrics.json"
    fine_metrics = single_label_report(
        [row["eunis_code"] for row in selected],
        [row["parsed_eunis_code"] for row in predictions],
        [row[0] for row in candidate_rows],
    )
    _write_json(metrics_path, {"overall": fine_metrics})
    run_manifest_path = run_dir / "dspark_manifest.json"
    _write_json(
        run_manifest_path,
        {
            "computation_commit": "c" * 40,
            "validation_commit": "d" * 40,
            "frozen_manifest_sha256": _sha256(frozen_manifest_path),
            "sample": selection,
            "run_scope": {
                "kind": "full-pilot",
                "processed_row_count": len(selected),
                "processed_sample_ids_sha256": selection["sample_ids_sha256"],
            },
            "outputs_sha256": {
                "dspark_predictions.parquet": _sha256(predictions_path),
                "dspark_metrics.json": _sha256(metrics_path),
            },
        },
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    _write_json(
        taxonomy_path,
        {
            "format_version": 1,
            "classifications": [
                {
                    "classification_release": "EUNIS-test-terrestrial-v1",
                    "classification_source": "https://example.test/terrestrial",
                    "source_sha256": "a" * 64,
                    "levels": [1, 2, 3],
                    "nodes": [
                        {"code": "A0", "parent_code": None, "level": 1},
                        {"code": "A1", "parent_code": "A0", "level": 2},
                        {"code": "A11", "parent_code": "A1", "level": 3},
                        {"code": "A12", "parent_code": "A1", "level": 3},
                    ],
                },
                {
                    "classification_release": "EUNIS-test-marine-v1",
                    "classification_source": "https://example.test/marine",
                    "source_sha256": "b" * 64,
                    "levels": [1, 2, 3, 4],
                    "nodes": [
                        {"code": "M", "parent_code": None, "level": 1},
                        {"code": "M1", "parent_code": "M", "level": 2},
                        {"code": "M11", "parent_code": "M1", "level": 3},
                        {"code": "B11", "parent_code": "M11", "level": 4},
                    ],
                },
            ],
        },
    )
    return {
        "predictions": predictions_path,
        "run_manifest": run_manifest_path,
        "sample": sample_path,
        "frozen_manifest": frozen_manifest_path,
        "candidate_csv": candidate_path,
        "taxonomy": taxonomy_path,
        "metrics": metrics_path,
        "run_dir": run_dir,
    }


def _score(paths: dict[str, Path], output: Path) -> dict[str, Any]:
    return score_hierarchical_run(
        predictions_path=paths["predictions"],
        run_manifest_path=paths["run_manifest"],
        frozen_sample_path=paths["sample"],
        frozen_manifest_path=paths["frozen_manifest"],
        candidate_csv_path=paths["candidate_csv"],
        taxonomy_path=paths["taxonomy"],
        output_dir=output,
        scoring_commit="e" * 40,
    )


def _rewrite_predictions(paths: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), paths["predictions"], compression="zstd")
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["outputs_sha256"]["dspark_predictions.parquet"] = _sha256(paths["predictions"])
    _write_json(paths["run_manifest"], manifest)


def test_hierarchical_run_writes_auditable_outputs_and_preserves_inputs(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    input_bytes = {path: path.read_bytes() for path in paths.values() if path.is_file()}
    output = tmp_path / "derived" / "hierarchy-v1"

    result = _score(paths, output)

    assert {path.name for path in output.iterdir()} == {
        "hierarchical_dataset_card.md",
        "hierarchical_manifest.json",
        "hierarchical_metrics.csv",
        "hierarchical_metrics.json",
    }
    report = json.loads((output / "hierarchical_metrics.json").read_text(encoding="utf-8"))
    row = next(item for item in report["rows"] if item["level"] == 1 and item["group"] == "overall")
    assert (row["n"], row["valid_count"], row["failed_count"], row["truncated_count"]) == (
        2,
        1,
        0,
        1,
    )
    assert row["accuracy"] == 0.5
    assert report["original_leaf"]["historical_metrics_match"] is True
    assert (
        report["original_leaf"]["overall"]
        == json.loads(paths["metrics"].read_text(encoding="utf-8"))["overall"]
    )
    card = (output / "hierarchical_dataset_card.md").read_text(encoding="utf-8")
    assert "Level | Group" in card
    assert "| 4 | overall | 0 | 2 | 0 | 0 | 0 | 0 | — | — | — | — |" in card
    manifest = json.loads((output / "hierarchical_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs_sha256"]["hierarchical_metrics.csv"] == _sha256(
        output / "hierarchical_metrics.csv"
    )
    assert result["files_sha256"]["hierarchical_dataset_card.md"] == _sha256(
        output / "hierarchical_dataset_card.md"
    )
    assert all(path.read_bytes() == data for path, data in input_bytes.items())


def test_hierarchical_run_scales_to_balanced_1000_row_samples(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path, sample_size=1000)
    input_bytes = {path: path.read_bytes() for path in paths.values() if path.is_file()}
    output = tmp_path / "derived" / "hierarchy-v1"

    result = _score(paths, output)

    report = json.loads((output / "hierarchical_metrics.json").read_text(encoding="utf-8"))
    assert result["sample_size"] == 1000
    assert report["provenance"]["decision_counts"] == {"no": 500, "yes": 500}
    assert report["original_leaf"]["historical_metrics_match"] is True
    for level in (1, 2, 3):
        for group in ("yes", "no"):
            row = next(
                item for item in report["rows"] if item["level"] == level and item["group"] == group
            )
            assert (row["n"], row["valid_count"], row["failed_count"], row["truncated_count"]) == (
                500,
                400,
                50,
                50,
            )
    level4 = {row["group"]: row for row in report["rows"] if row["level"] == 4}
    assert (level4["yes"]["n"], level4["yes"]["excluded_shallower_reference_count"]) == (
        5,
        495,
    )
    assert (level4["no"]["n"], level4["no"]["excluded_shallower_reference_count"]) == (
        5,
        495,
    )
    assert (
        level4["overall"]["n"],
        level4["overall"]["excluded_shallower_reference_count"],
    ) == (10, 990)
    assert level4["overall"]["accuracy"] == 1
    assert level4["overall"]["multiclass_mcc"] == 0
    assert all(path.read_bytes() == data for path, data in input_bytes.items())


def test_hierarchical_run_rejects_smoke_scope_and_existing_output(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    run_manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    run_manifest["run_scope"]["kind"] = "bounded-smoke"
    _write_json(paths["run_manifest"], run_manifest)
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="requires a full pilot run"):
        _score(paths, output)

    _write_json(paths["run_manifest"], {"bad": "manifest"})
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _score(paths, output)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_outputs", "frozen sample manifest has no output checksums"),
        ("checksum", "frozen sample output checksum mismatch"),
        ("sample_ids", "frozen sample manifest ID digest mismatch"),
    ],
)
def test_hierarchical_run_rejects_frozen_manifest_mismatch(
    change: str, message: str, tmp_path: Path
) -> None:
    paths = _fixtures(tmp_path)
    manifest = json.loads(paths["frozen_manifest"].read_text(encoding="utf-8"))
    if change == "missing_outputs":
        manifest["outputs_sha256"] = None
    elif change == "checksum":
        manifest["outputs_sha256"]["candidate_labels.csv"] = "bad"
    else:
        manifest["sample_ids_sha256"] = "bad"
    _write_json(paths["frozen_manifest"], manifest)

    with pytest.raises(ValueError, match=message):
        _score(paths, tmp_path / "derived")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_id", "unknown-sample", "prediction rows do not match frozen sample order"),
        ("decision", "maybe", "prediction row does not match frozen decision"),
        ("eunis_code", "X", "prediction row does not match frozen eunis_code"),
        (
            "eunis_source_version",
            "different map",
            "prediction row does not match frozen eunis_source_version",
        ),
    ],
)
def test_hierarchical_run_rejects_prediction_identity_mismatch(
    field: str, value: str, message: str, tmp_path: Path
) -> None:
    paths = _fixtures(tmp_path)
    rows = pq.read_table(paths["predictions"]).to_pylist()
    rows[0][field] = value
    _rewrite_predictions(paths, rows)

    with pytest.raises(ValueError, match=message):
        _score(paths, tmp_path / "derived")


def test_score_hierarchy_cli_routes_all_artifact_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}

    def score(**kwargs: Any) -> dict[str, str]:
        observed.update(kwargs)
        return {"output_dir": "derived"}

    monkeypatch.setattr(cli, "score_hierarchical_run", score)
    assert (
        cli.main(
            [
                "score-hierarchy",
                "--predictions",
                "run/dspark_predictions.parquet",
                "--run-manifest",
                "run/dspark_manifest.json",
                "--frozen-sample",
                "frozen/frozen_sample.json",
                "--frozen-manifest",
                "frozen/manifest.json",
                "--candidate-csv",
                "frozen/candidate_labels.csv",
                "--taxonomy",
                "taxonomy.json",
                "--output-dir",
                "derived",
                "--scoring-commit",
                "e" * 40,
            ]
        )
        == 0
    )

    assert observed["predictions_path"] == Path("run/dspark_predictions.parquet")
    assert observed["run_manifest_path"] == Path("run/dspark_manifest.json")
    assert observed["frozen_sample_path"] == Path("frozen/frozen_sample.json")
    assert observed["taxonomy_path"] == Path("taxonomy.json")
    assert observed["scoring_commit"] == "e" * 40
    assert json.loads(capsys.readouterr().out) == {"output_dir": "derived"}
