"""Retrospective scoring publishes verified derived files without touching its source run."""

from __future__ import annotations

import codecs
import csv
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset_text_label_benchmark.pilot import cli, hierarchy_runner
from georeset_text_label_benchmark.pilot.hierarchy_runner import score_hierarchical_run
from georeset_text_label_benchmark.pilot.metrics import single_label_report


@contextmanager
def _raises_exact(expected_type: type[Exception], message: str) -> Iterator[None]:
    with pytest.raises(expected_type) as caught:
        yield
    assert str(caught.value) == message


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


def _score(paths: dict[str, Path], output: Path, scoring_commit: str = "e" * 40) -> dict[str, Any]:
    return score_hierarchical_run(
        predictions_path=paths["predictions"],
        run_manifest_path=paths["run_manifest"],
        frozen_sample_path=paths["sample"],
        frozen_manifest_path=paths["frozen_manifest"],
        candidate_csv_path=paths["candidate_csv"],
        taxonomy_path=paths["taxonomy"],
        output_dir=output,
        scoring_commit=scoring_commit,
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
    sample = json.loads(paths["sample"].read_text(encoding="utf-8"))
    input_hashes = {
        "dspark_predictions.parquet": _sha256(paths["predictions"]),
        "dspark_manifest.json": _sha256(paths["run_manifest"]),
        "dspark_metrics.json": _sha256(paths["metrics"]),
        "frozen_sample.json": _sha256(paths["sample"]),
        "frozen_manifest.json": _sha256(paths["frozen_manifest"]),
        "candidate_labels.csv": _sha256(paths["candidate_csv"]),
        "taxonomy_mapping.json": _sha256(paths["taxonomy"]),
    }
    assert report["provenance"] == {
        "scoring_protocol": "eunis-hierarchy-v1",
        "scoring_commit": "e" * 40,
        "historical_computation_commit": "c" * 40,
        "historical_validation_commit": "d" * 40,
        "sample_ids_sha256": sample["selection"]["sample_ids_sha256"],
        "sample_size": 2,
        "decision_counts": {"no": 1, "yes": 1},
        "inputs_sha256": input_hashes,
        "taxonomy_classifications": report["classifications"],
    }
    assert (
        report["original_leaf"]["historical_metrics_sha256"] == input_hashes["dspark_metrics.json"]
    )
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
    output_hashes = {
        name: _sha256(output / name)
        for name in (
            "hierarchical_metrics.json",
            "hierarchical_metrics.csv",
            "hierarchical_dataset_card.md",
            "hierarchical_manifest.json",
        )
    }
    assert manifest == {
        "format_version": 1,
        "protocol_version": "eunis-hierarchy-v1",
        "scoring_commit": "e" * 40,
        "historical_computation_commit": "c" * 40,
        "historical_validation_commit": "d" * 40,
        "sample_ids_sha256": sample["selection"]["sample_ids_sha256"],
        "input_sha256": input_hashes,
        "taxonomy_classifications": report["classifications"],
        "outputs_sha256": {
            name: output_hashes[name]
            for name in (
                "hierarchical_metrics.json",
                "hierarchical_metrics.csv",
                "hierarchical_dataset_card.md",
            )
        },
    }
    assert result == {
        "output_dir": str(output),
        "sample_size": 2,
        "classification_count": 2,
        "level_group_rows": 12,
        "files_sha256": output_hashes,
    }
    csv_lines = (output / "hierarchical_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == (
        "level,group,sample_n,eligible_n,n,excluded_shallower_reference_count,valid_count,"
        "failed_count,truncated_count,unmapped_count,unmapped_gold_count,"
        "unmapped_prediction_count,correct_count,accuracy,valid_mapped_count,"
        "valid_only_accuracy,macro_f1,multiclass_mcc,label_universe"
    )
    assert len(csv_lines) == 13
    assert "\r" not in (output / "hierarchical_metrics.csv").read_text(encoding="utf-8")
    assert all(path.read_bytes() == data for path, data in input_bytes.items())


def test_hierarchical_run_rejects_an_invalid_scoring_commit_before_io(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    output = tmp_path / "derived" / "hierarchy-v1"

    with _raises_exact(
        ValueError, "scoring_commit must be a 40-character lowercase Git commit SHA"
    ):
        _score(paths, output, scoring_commit="invalid")

    assert not output.exists()


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

    with _raises_exact(ValueError, "hierarchical retrospective scoring requires a full pilot run"):
        _score(paths, output)

    _write_json(paths["run_manifest"], {"bad": "manifest"})
    output.mkdir()
    with _raises_exact(FileExistsError, f"hierarchy output directory already exists: {output}"):
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

    with _raises_exact(ValueError, message):
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

    with _raises_exact(ValueError, message):
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

    assert observed == {
        "predictions_path": Path("run/dspark_predictions.parquet"),
        "run_manifest_path": Path("run/dspark_manifest.json"),
        "frozen_sample_path": Path("frozen/frozen_sample.json"),
        "frozen_manifest_path": Path("frozen/manifest.json"),
        "candidate_csv_path": Path("frozen/candidate_labels.csv"),
        "taxonomy_path": Path("taxonomy.json"),
        "output_dir": Path("derived"),
        "scoring_commit": "e" * 40,
    }
    assert json.loads(capsys.readouterr().out) == {"output_dir": "derived"}


def test_score_hierarchy_parser_requires_all_inputs_and_pins_help_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._parser()
    with pytest.raises(SystemExit) as help_result:
        parser.parse_args(["score-hierarchy", "--help"])
    assert help_result.value.code == 0
    assert capsys.readouterr().out == (
        "usage: georeset-pilot score-hierarchy [-h] --predictions PREDICTIONS\n"
        "                                      --run-manifest RUN_MANIFEST\n"
        "                                      --frozen-sample FROZEN_SAMPLE\n"
        "                                      --frozen-manifest FROZEN_MANIFEST\n"
        "                                      --candidate-csv CANDIDATE_CSV --taxonomy\n"
        "                                      TAXONOMY --output-dir OUTPUT_DIR\n"
        "                                      --scoring-commit SCORING_COMMIT\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --predictions PREDICTIONS\n"
        "  --run-manifest RUN_MANIFEST\n"
        "  --frozen-sample FROZEN_SAMPLE\n"
        "  --frozen-manifest FROZEN_MANIFEST\n"
        "  --candidate-csv CANDIDATE_CSV\n"
        "  --taxonomy TAXONOMY\n"
        "  --output-dir OUTPUT_DIR\n"
        "  --scoring-commit SCORING_COMMIT\n"
    )
    required = [
        ("--predictions", "predictions.parquet"),
        ("--run-manifest", "run.json"),
        ("--frozen-sample", "sample.json"),
        ("--frozen-manifest", "frozen.json"),
        ("--candidate-csv", "candidates.csv"),
        ("--taxonomy", "taxonomy.json"),
        ("--output-dir", "derived"),
        ("--scoring-commit", "e" * 40),
    ]
    base = [item for pair in required for item in pair]
    for missing in required:
        argv = ["score-hierarchy"] + [
            token for pair in required if pair != missing for token in pair
        ]
        with pytest.raises(SystemExit) as error:
            parser.parse_args(argv)
        assert error.value.code == 2
    with pytest.raises(SystemExit) as no_command:
        parser.parse_args([])
    assert no_command.value.code == 2
    assert len(base) == 16


def test_hierarchy_cli_prints_non_ascii_results_without_escaping(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = {"habitat": "湿地 - forêt"}
    monkeypatch.setattr(cli, "_freeze_sample_command", lambda args: expected)
    assert (
        cli.main(
            [
                "freeze",
                "--labelled-parquet",
                "pool.parquet",
                "--pipeline-manifest",
                "manifest.json",
            ]
        )
        == 0
    )
    assert (
        capsys.readouterr().out
        == json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def test_hierarchy_dataset_card_matches_the_complete_protocol_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card_path = tmp_path / "hierarchical_dataset_card.md"
    original_open = Path.open
    open_arguments: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def tracked_open(target: Path, *args: Any, **kwargs: Any):
        if target == card_path:
            open_arguments.append((args, kwargs))
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    report = {
        "rows": [
            {
                "level": 3,
                "group": "overall",
                "n": 2,
                "excluded_shallower_reference_count": 1,
                "valid_count": 1,
                "failed_count": 0,
                "truncated_count": 1,
                "unmapped_count": 1,
                "accuracy": 0.5,
                "valid_only_accuracy": 0.5,
                "macro_f1": 0.25,
                "multiclass_mcc": 0.75,
            }
        ],
        "original_leaf": {
            "overall": {
                "sample_count": 2,
                "prediction_count": 1,
                "invalid_output_count": 1,
                "top1_accuracy": 0.5,
                "coverage": 0.5,
                "macro_f1_all_candidates": 0.25,
            },
            "by_decision": [
                {
                    "group": "yes",
                    "sample_count": 1,
                    "prediction_count": 1,
                    "top1_accuracy": 1.0,
                    "coverage": 1.0,
                },
                {
                    "group": "no",
                    "sample_count": 1,
                    "prediction_count": 0,
                    "top1_accuracy": 0.0,
                    "coverage": 0.0,
                },
            ],
        },
    }
    hierarchy_runner._write_dataset_card(card_path, report)
    assert len(open_arguments) == 1
    args, kwargs = open_arguments[0]
    assert args == ("x",)
    assert kwargs == {"encoding": "utf-8", "newline": "\n"}

    assert card_path.read_text(encoding="utf-8") == (
        "## Retrospective hierarchical EUNIS evaluation\n\n"
        "The original EUNIS leaf predictions and reference labels are unchanged. Levels 1 through "
        "3 include every row; level 4 includes only references with a level 4 target, and reports "
        "shallower exclusions explicitly. Invalid and truncated predictions remain in each eligible "
        "denominator.\n\n"
        "### Official ancestor levels\n\n"
        "| Level | Group | n | Excluded shallower | Valid | Failed | Truncated | Unmapped | Accuracy | "
        "Valid-only accuracy | Macro-F1 | Multiclass MCC |\n"
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        "| 3 | overall | 2 | 1 | 1 | 0 | 1 | 1 | 0.5000 | 0.5000 | 0.2500 | 0.7500 |\n\n"
        "### Original leaf-label results\n\n"
        "| Group | n | Valid predictions | Invalid outputs | Accuracy | Coverage | Macro-F1 |\n"
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        "| overall | 2 | 1 | 1 | 0.5000 | 0.5000 | 0.2500 |\n"
        "| yes | 1 | 1 | 0 | 1.0000 | 1.0000 | — |\n"
        "| no | 1 | 0 | 1 | 0.0000 | 0.0000 | — |\n\n"
        "Ancestor scoring uses the source-specific official EEA terrestrial 2021 and marine 2022 "
        "mappings. Label universes and input checksums are in `hierarchical_metrics.json` and "
        "`hierarchical_manifest.json`.\n"
    )


def test_candidate_metadata_and_csv_validation_reject_incomplete_inputs_exactly() -> None:
    with _raises_exact(ValueError, "candidate taxonomy provenance is incomplete: A11"):
        hierarchy_runner._candidate_metadata(
            {
                "classification_release": "",
                "classification_source": "https://example.test/source",
                "source_sha256": "a" * 64,
            },
            "A11",
        )

    fields = [
        "eunis_code",
        "classification_release",
        "classification_source",
        "source_sha256",
    ]
    with _raises_exact(ValueError, "candidate CSV codes must be non-empty and unique"):
        hierarchy_runner._candidate_sources_from_rows(
            fields,
            [{"eunis_code": "", "classification_release": "x"}],
            {"codes": []},
        )
    with _raises_exact(ValueError, "candidate taxonomy provenance is incomplete: A11"):
        hierarchy_runner._candidate_sources_from_rows(
            fields,
            [
                {
                    "eunis_code": "A11",
                    "classification_release": "",
                    "classification_source": "source",
                    "source_sha256": "a" * 64,
                }
            ],
            {"codes": ["A11"]},
        )
    with _raises_exact(ValueError, "candidate CSV codes must be non-empty and unique"):
        hierarchy_runner._candidate_sources_from_rows(
            fields,
            [
                {
                    "eunis_code": "A11",
                    "classification_release": "r",
                    "classification_source": "source",
                    "source_sha256": "a" * 64,
                },
                {
                    "eunis_code": "A11",
                    "classification_release": "r",
                    "classification_source": "source",
                    "source_sha256": "a" * 64,
                },
            ],
            {"codes": ["A11"]},
        )
    with _raises_exact(ValueError, "candidate CSV code list does not match the frozen sample"):
        hierarchy_runner._candidate_sources_from_rows(
            fields,
            [
                {
                    "eunis_code": "A12",
                    "classification_release": "release",
                    "classification_source": "source",
                    "source_sha256": "a" * 64,
                }
            ],
            {"codes": ["A11"]},
        )
    with _raises_exact(ValueError, "candidate CSV is missing classification provenance columns"):
        hierarchy_runner._validate_candidate_csv_fields(None)
    with _raises_exact(ValueError, "candidate CSV is missing classification provenance columns"):
        hierarchy_runner._validate_candidate_csv_fields(["eunis_code"])


def test_candidate_csv_reader_checks_checksum_decoding_and_failure_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate_labels.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
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
        writer.writerow(
            {
                "eunis_code": "A11",
                "classification_release": "EUNIS-é",
                "classification_source": "https://example.test/湿地",
                "source_sha256": "a" * 64,
            }
        )
    digest = _sha256(path)
    sample = {"candidate_labels": {"codes": ["A11"], "sha256": digest}}
    original_open = Path.open
    open_arguments: list[dict[str, Any]] = []

    def tracked_open(target: Path, *args: Any, **kwargs: Any):
        if target == path:
            open_arguments.append(kwargs)
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    assert hierarchy_runner._read_candidate_sources(
        path, sample, {"candidate_labels.csv": digest}
    ) == {
        "A11": {
            "classification_release": "EUNIS-é",
            "classification_source": "https://example.test/湿地",
            "source_sha256": "a" * 64,
        }
    }
    assert len(open_arguments) == 1
    encoding = open_arguments[0].get("encoding")
    assert isinstance(encoding, str)
    assert codecs.lookup(encoding).name == "utf-8"
    assert open_arguments[0].get("newline") == ""
    with _raises_exact(ValueError, "candidate CSV checksum does not match the frozen sample"):
        hierarchy_runner._read_candidate_sources(
            path,
            {"candidate_labels": {"codes": ["A11"], "sha256": "wrong"}},
            {"candidate_labels.csv": digest},
        )
    with _raises_exact(ValueError, "candidate CSV checksum does not match the frozen sample"):
        hierarchy_runner._read_candidate_sources(
            path, {"candidate_labels": None}, {"candidate_labels.csv": digest}
        )

    missing = tmp_path / "missing-candidates.csv"
    with _raises_exact(ValueError, f"cannot read candidate CSV: {missing}"):
        hierarchy_runner._read_candidate_sources(
            missing,
            {"candidate_labels": {"codes": [], "sha256": "expected"}},
            {"candidate_labels.csv": "expected"},
        )


def test_json_reader_enforces_utf8_objects_and_failure_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "unicode.json"
    json_path.write_text('{"label": "湿地"}', encoding="utf-8")
    original_read_text = Path.read_text
    encodings: list[str | None] = []

    def tracked_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == json_path:
            encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read_text)
    assert hierarchy_runner._read_json(json_path) == {"label": "湿地"}
    assert len(encodings) == 1
    encoding = encodings[0]
    assert encoding is not None
    assert codecs.lookup(encoding).name == "utf-8"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    with _raises_exact(ValueError, f"cannot read JSON object: {bad_json}"):
        hierarchy_runner._read_json(bad_json)
    non_object_json = tmp_path / "array.json"
    non_object_json.write_text("[]", encoding="utf-8")
    with _raises_exact(ValueError, f"JSON artifact must be an object: {non_object_json}"):
        hierarchy_runner._read_json(non_object_json)


def test_prediction_reader_rejects_empty_non_object_and_unreadable_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction_path = tmp_path / "predictions.parquet"

    class Table:
        def __init__(self, rows: list[Any]) -> None:
            self.rows = rows

        def to_pylist(self) -> list[Any]:
            return self.rows

    monkeypatch.setattr(hierarchy_runner.pq, "read_table", lambda path: Table([]))
    with _raises_exact(ValueError, "historical prediction Parquet must contain row objects"):
        hierarchy_runner._read_predictions(prediction_path)
    monkeypatch.setattr(hierarchy_runner.pq, "read_table", lambda path: Table([None]))
    with _raises_exact(ValueError, "historical prediction Parquet must contain row objects"):
        hierarchy_runner._read_predictions(prediction_path)

    def read_failure(path: Path) -> None:
        raise OSError("unreadable")

    monkeypatch.setattr(hierarchy_runner.pq, "read_table", read_failure)
    with _raises_exact(ValueError, f"cannot read prediction Parquet: {prediction_path}"):
        hierarchy_runner._read_predictions(prediction_path)


def test_frozen_sample_and_candidate_helpers_validate_shapes_and_ids() -> None:
    with _raises_exact(ValueError, "frozen sample must contain selected rows"):
        hierarchy_runner._sample_rows({"selected_rows": []})
    with _raises_exact(ValueError, "frozen sample must contain selected rows"):
        hierarchy_runner._sample_rows({"selected_rows": [None]})
    with _raises_exact(ValueError, "frozen sample IDs must be unique non-empty strings"):
        hierarchy_runner._sample_ids([{"sample_id": ""}])
    with _raises_exact(ValueError, "frozen sample IDs must be unique non-empty strings"):
        hierarchy_runner._sample_ids([{"sample_id": "same"}, {"sample_id": "same"}])
    with _raises_exact(ValueError, "frozen sample IDs must be unique non-empty strings"):
        hierarchy_runner._sample_ids([{"sample_id": 7}])


def test_source_artifact_validators_check_scope_hashes_and_metadata() -> None:
    selection = {"sample_size": 1, "sample_ids_sha256": "expected", "seed": 42}
    with _raises_exact(ValueError, "hierarchical retrospective scoring requires a full pilot run"):
        hierarchy_runner._validate_full_pilot_scope(
            {
                "run_scope": {
                    "kind": "bounded-smoke",
                    "processed_row_count": 1,
                    "processed_sample_ids_sha256": "expected",
                }
            },
            selection,
            row_count=1,
        )
    for scope in (
        {"kind": "full-pilot", "processed_row_count": 2, "processed_sample_ids_sha256": "expected"},
        {"kind": "full-pilot", "processed_row_count": 1, "processed_sample_ids_sha256": "other"},
    ):
        with _raises_exact(
            ValueError, "DSpark manifest run scope does not match the frozen sample"
        ):
            hierarchy_runner._validate_full_pilot_scope(
                {"run_scope": scope}, selection, row_count=1
            )

    hashes = {"dspark_predictions.parquet": "predictions", "dspark_metrics.json": "metrics"}
    with _raises_exact(ValueError, "DSpark manifest has no output checksums"):
        hierarchy_runner._validate_run_output_hashes({}, hashes)
    with _raises_exact(ValueError, "DSpark output checksum mismatch"):
        hierarchy_runner._validate_run_output_hashes(
            {"outputs_sha256": {"dspark_predictions.parquet": "wrong"}}, hashes
        )
    with _raises_exact(ValueError, "DSpark manifest points to a different frozen sample manifest"):
        hierarchy_runner._validate_run_sample_reference(
            {"frozen_manifest_sha256": "wrong", "sample": selection},
            {"frozen_manifest.json": "expected"},
            selection,
        )
    with _raises_exact(ValueError, "DSpark manifest sample metadata does not match frozen input"):
        hierarchy_runner._validate_run_sample_reference(
            {"frozen_manifest_sha256": "expected", "sample": {}},
            {"frozen_manifest.json": "expected"},
            selection,
        )

    with _raises_exact(ValueError, "frozen sample ID digest is inconsistent"):
        hierarchy_runner._validate_source_artifacts(
            {
                "selected_rows": [{"sample_id": "id"}],
                "selection": {"sample_ids_sha256": "wrong"},
            },
            {},
            {},
            {},
            [],
        )


def test_original_leaf_metrics_require_exact_historical_values() -> None:
    values = {
        "sample_count": 2,
        "candidate_class_count": 158,
        "prediction_count": 1,
        "invalid_output_count": 1,
        "coverage": 0.5,
        "top1_accuracy": 0.5,
        "macro_f1_all_candidates": 0.25,
    }
    report = {"original_leaf": {"overall": values}}
    with _raises_exact(ValueError, "historical DSpark metrics have no overall leaf report"):
        hierarchy_runner._validate_original_leaf_metrics(report, {})
    for field, value in values.items():
        mismatched = {**values, field: value + 1}
        with _raises_exact(
            ValueError, "recomputed original leaf metrics differ from historical DSpark metrics"
        ):
            hierarchy_runner._validate_original_leaf_metrics(report, {"overall": mismatched})


def test_input_files_reject_symlinks_and_missing_paths_exactly(tmp_path: Path) -> None:
    regular = tmp_path / "regular.json"
    regular.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(regular)
    missing = tmp_path / "missing.json"
    with _raises_exact(
        FileNotFoundError,
        f"required historical input is not a regular file: {symlink}",
    ):
        hierarchy_runner._validate_input_files({"source": symlink})
    with _raises_exact(
        FileNotFoundError,
        f"required historical input is not a regular file: {missing}",
    ):
        hierarchy_runner._validate_input_files({"source": missing})


def test_prediction_vector_helpers_reject_misalignment_and_normalize_bad_values() -> None:
    sample_rows = [
        {"decision": "yes", "eunis_code": "A11", "eunis_source_version": "v1"},
        {"decision": "no", "eunis_code": "A12", "eunis_source_version": "v1"},
    ]
    with pytest.raises(ValueError, match=r"^zip\(\) argument"):
        hierarchy_runner._row_vectors(sample_rows, [{"parse_status": "valid"}])
    assert hierarchy_runner._prediction_code({"parsed_eunis_code": 7}) is None
    assert hierarchy_runner._prediction_code({"parsed_eunis_code": "A11"}) == "A11"
    assert hierarchy_runner._prediction_status({"parse_status": 7}) == "failed"
    assert hierarchy_runner._prediction_status({"parse_status": "truncated"}) == "truncated"


def test_failed_output_writes_keep_the_named_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "derived" / "hierarchy-v1"

    def fail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        raise OSError("table write failed")

    monkeypatch.setattr(hierarchy_runner, "_write_csv", fail_csv)
    with pytest.raises(OSError, match="table write failed"):
        hierarchy_runner._write_outputs(
            output,
            {"rows": [], "classifications": []},
            {},
            {"selection": {"sample_size": 0, "sample_ids_sha256": "id"}},
            {},
            "e" * 40,
        )
    assert not output.exists()
    staging = list(output.parent.glob(".hierarchy-v1.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / "hierarchical_metrics.json").is_file()


def test_hierarchy_csv_uses_exact_fields_compact_json_and_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "hierarchical_metrics.csv"
    row = {
        "level": 1,
        "group": "overall",
        "sample_n": 1,
        "eligible_n": 1,
        "n": 1,
        "excluded_shallower_reference_count": 0,
        "valid_count": 1,
        "failed_count": 0,
        "truncated_count": 0,
        "unmapped_count": 0,
        "unmapped_gold_count": 0,
        "unmapped_prediction_count": 0,
        "correct_count": 1,
        "accuracy": 1.0,
        "valid_mapped_count": 1,
        "valid_only_accuracy": 1.0,
        "macro_f1": 1.0,
        "multiclass_mcc": 1.0,
        "label_universe": ["é:A0", "湿地:B0"],
    }
    original_open = Path.open
    open_arguments: list[dict[str, Any]] = []

    def tracked_open(target: Path, *args: Any, **kwargs: Any):
        if target == path and args and args[0] == "x":
            open_arguments.append(kwargs)
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    missing_metrics_row = {
        **row,
        "group": "no",
        "accuracy": None,
        "valid_only_accuracy": None,
        "macro_f1": None,
        "multiclass_mcc": None,
        "label_universe": [],
    }
    hierarchy_runner._write_csv(path, [row, missing_metrics_row])
    content = path.read_bytes().decode("utf-8")
    lines = content.splitlines()
    assert lines[0] == (
        "level,group,sample_n,eligible_n,n,excluded_shallower_reference_count,valid_count,"
        "failed_count,truncated_count,unmapped_count,unmapped_gold_count,"
        "unmapped_prediction_count,correct_count,accuracy,valid_mapped_count,"
        "valid_only_accuracy,macro_f1,multiclass_mcc,label_universe"
    )
    with path.open(encoding="utf-8", newline="") as stream:
        parsed = list(csv.DictReader(stream))
    assert len(parsed) == 2
    assert parsed[0]["group"] == "overall"
    assert parsed[0]["label_universe"] == '["é:A0","湿地:B0"]'
    assert parsed[0]["accuracy"] == "1.0"
    assert parsed[1]["group"] == "no"
    assert parsed[1]["accuracy"] == ""
    assert parsed[1]["macro_f1"] == ""
    assert len(open_arguments) == 1
    encoding = open_arguments[0].get("encoding")
    assert isinstance(encoding, str)
    assert codecs.lookup(encoding).name == "utf-8"
    assert open_arguments[0].get("newline") == ""
    assert content.endswith("\n")
    assert "\r" not in content
    assert len(lines) == 3
